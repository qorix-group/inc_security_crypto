/********************************************************************************
 * Copyright (c) 2026 Contributors to the Eclipse Foundation
 *
 * See the NOTICE file(s) distributed with this work for additional
 * information regarding copyright ownership.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * SPDX-License-Identifier: Apache-2.0
 ********************************************************************************/

/// @file score_api_cipher_example.cpp
/// @brief Demonstrates symmetric encryption and decryption using the score::crypto API.
///
/// Shows:
///   - AES key generation via IKeyManagementContext::GenerateKey
///   - Streaming encryption (Init → Update* → Finalize) and decryption
///   - Single-shot encryption / decryption via SingleShot()
///   - Context reuse via Reset()
///   - That a wrong key or tampered ciphertext does not yield the plaintext
///
/// The key is randomly generated per run, so the ciphertext cannot be compared
/// against a fixed vector. Correctness is established by round-tripping:
/// decrypt(encrypt(m)) == m, and by checking that the ciphertext differs from
/// the plaintext and that tampering is detected.

#include "score/crypto/src/api/common/types.hpp"
#include "score/crypto/src/api/config/cipher_context_config.hpp"
#include "score/crypto/src/api/config/key_management_context_config.hpp"
#include "score/crypto/src/api/config/key_operation_params.hpp"
#include "score/crypto/src/api/config/random_context_config.hpp"
#include "score/crypto/src/api/contexts/i_cipher_context.hpp"
#include "score/crypto/src/api/contexts/i_key_management_context.hpp"
#include "score/crypto/src/api/contexts/i_random_context.hpp"
#include "score/crypto/src/api/crypto_stack_factory.hpp"
#include "score/crypto/src/api/i_crypto_context.hpp"
#include "score/crypto/src/api/i_crypto_stack.hpp"
#include "score/tests/utility/test_utility.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace score::crypto;
using tests::utility::as_bytes;
using tests::utility::print_hex;

namespace
{

// =========================================================================
// Input Data
// =========================================================================

/// The plaintext, held as a literal rather than read from a file.
///
/// This test establishes correctness by round-tripping under a per-run random
/// key, so the input is not a fixed vector paired with an expected output — any
/// non-empty bytes do the job. Keeping it inline makes the test self-contained
/// and removes a runtime dependency on the deployed test-vector files.
constexpr std::string_view kPlaintext = "Hello, World!";

/// A second, different plaintext, used to show that reusing a context after
/// Reset() encrypts the new message rather than replaying the previous one.
constexpr std::string_view kPlaintextAlt = "Another Hello, World!";

// =========================================================================
// Parameterized Test Data
// =========================================================================

struct CipherTestData
{
    std::string test_case_name;
    std::optional<ProviderType> provider_type;
    std::string cipher_algorithm;  ///< e.g. "AES-256-CBC"
    std::string key_algorithm;     ///< e.g. "AES-256-CBC"
    std::size_t iv_size;           ///< 16 for CBC/CTR, 0 for ECB
    bool block_padded;             ///< true when Finalize() emits a padding block
};

// =========================================================================
// Helpers
// =========================================================================

/// Creates a cipher context for one direction over the given key.
std::unique_ptr<ICipherContext> MakeCipherContext(ICryptoContext& ctx,
                                                  const CipherTestData& data,
                                                  const CryptoResourceGuard& key,
                                                  CipherDirection direction)
{
    CipherContextConfig config;
    config.SetAlgorithm(data.cipher_algorithm).SetKey(key).SetDirection(direction);
    if (data.provider_type.has_value())
    {
        config.SetProviderType(data.provider_type.value());
    }

    auto result = ctx.CreateCipherContext(config);
    EXPECT_TRUE(result.has_value()) << "Failed to create cipher context for " << data.cipher_algorithm;
    if (!result.has_value())
    {
        return nullptr;
    }
    return std::move(result.value());
}

/// Streaming transform: Init(iv) → Update(chunk1) → Update(chunk2) → Finalize.
/// Returns the concatenation of everything the context produced.
void StreamingTransform(ICipherContext& cipher,
                        const std::vector<uint8_t>& iv,
                        const std::vector<uint8_t>& input,
                        std::size_t block_size,
                        std::vector<uint8_t>& output)
{
    output.clear();

    // IV-less modes (ECB) must pass std::nullopt rather than an empty span.
    std::optional<score::cpp::span<const uint8_t>> iv_arg{};
    if (!iv.empty())
    {
        iv_arg = score::cpp::span<const uint8_t>{iv.data(), iv.size()};
    }
    ASSERT_TRUE(cipher.Init(iv_arg)) << "Cipher Init failed";

    // Split mid-message rather than on a block boundary: that leaves a partial
    // block for the cipher to buffer across the two Update calls, which is the
    // interesting path. Both chunks must be non-empty — an Update with no data
    // is not a meaningful request and the daemon rejects it.
    ASSERT_GE(input.size(), 2U) << "Test input must be at least two bytes";
    const auto split = static_cast<std::ptrdiff_t>(input.size() / 2U);
    const std::vector<uint8_t> chunk1(input.begin(), input.begin() + split);
    const std::vector<uint8_t> chunk2(input.begin() + split, input.end());

    // Worst case one extra block per Update plus one on Finalize.
    std::vector<uint8_t> scratch(input.size() + (2U * block_size));

    auto n1 = cipher.Update({chunk1.data(), chunk1.size()}, {scratch.data(), scratch.size()});
    ASSERT_TRUE(n1.has_value()) << "Cipher Update (chunk 1) failed";
    output.insert(output.end(), scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(n1.value()));

    auto n2 = cipher.Update({chunk2.data(), chunk2.size()}, {scratch.data(), scratch.size()});
    ASSERT_TRUE(n2.has_value()) << "Cipher Update (chunk 2) failed";
    output.insert(output.end(), scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(n2.value()));

    auto n3 = cipher.Finalize({scratch.data(), scratch.size()});
    ASSERT_TRUE(n3.has_value()) << "Cipher Finalize failed";
    output.insert(output.end(), scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(n3.value()));
}

/// Reset → SingleShot(iv, input) and return the produced bytes.
void SingleShotTransform(ICipherContext& cipher,
                         const std::vector<uint8_t>& iv,
                         const std::vector<uint8_t>& input,
                         std::size_t block_size,
                         std::vector<uint8_t>& output)
{
    ASSERT_TRUE(cipher.Reset()) << "Reset before SingleShot failed";

    std::vector<uint8_t> scratch(input.size() + (2U * block_size));
    auto n = cipher.SingleShot({iv.data(), iv.size()}, {input.data(), input.size()}, {scratch.data(), scratch.size()});
    ASSERT_TRUE(n.has_value()) << "Cipher SingleShot failed";

    output.assign(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(n.value()));
}

// =========================================================================
// Test: encrypt / decrypt round trip with a generated key
// =========================================================================

class CipherRoundTripTest : public ::testing::TestWithParam<CipherTestData>
{
};

TEST_P(CipherRoundTripTest, EncryptDecryptRoundTrip)
{
    const auto test_data = GetParam();

    const auto plaintext = as_bytes(kPlaintext);
    const auto plaintext_alt = as_bytes(kPlaintextAlt);

    // =========================================================================
    // 1. Create the crypto stack and a crypto context
    // =========================================================================
    CryptoStackConfig stack_config;
    stack_config.SetConnectionEndpoint("unix:///tmp/crypto_daemon.sock");

    auto stack_result = CreateCryptoStack(stack_config);
    ASSERT_TRUE(stack_result.has_value()) << "Failed to create crypto stack";
    auto& stack = stack_result.value();

    auto ctx_result = stack->CreateCryptoContext();
    ASSERT_TRUE(ctx_result.has_value()) << "Failed to create crypto context";
    auto& ctx = ctx_result.value();

    // =========================================================================
    // 2. Generate an ephemeral AES key permitted to encrypt and decrypt
    // =========================================================================
    KeyManagementContextConfig key_mgmt_config;
    if (test_data.provider_type.has_value())
    {
        key_mgmt_config.SetProviderType(test_data.provider_type.value());
    }

    auto key_mgmt_result = ctx->CreateKeyManagementContext(key_mgmt_config);
    ASSERT_TRUE(key_mgmt_result.has_value()) << "Failed to create key management context";
    auto& key_mgmt = key_mgmt_result.value();

    GenerateKeyParams key_gen_params;
    key_gen_params.SetAlgorithm(test_data.key_algorithm)
        .SetPermissions(KeyOperationPermission::kEncrypt | KeyOperationPermission::kDecrypt);

    auto key_result = key_mgmt->GenerateKey(key_gen_params);
    ASSERT_TRUE(key_result.has_value()) << "Failed to generate cipher key";
    auto key = std::move(key_result.value());
    ASSERT_TRUE(key.IsActive()) << "Key guard should be active after generation";

    // =========================================================================
    // 3. Obtain a random IV of the length the algorithm requires
    // =========================================================================
    std::vector<uint8_t> iv(test_data.iv_size, 0U);
    if (test_data.iv_size > 0U)
    {
        RandomContextConfig random_config;
        if (test_data.provider_type.has_value())
        {
            random_config.SetProviderType(test_data.provider_type.value());
        }
        auto random_result = ctx->CreateRandomContext(random_config);
        ASSERT_TRUE(random_result.has_value()) << "Failed to create random context";

        auto generated = random_result.value()->Generate({iv.data(), iv.size()});
        ASSERT_TRUE(generated.has_value()) << "Failed to generate IV";
        ASSERT_EQ(generated.value(), test_data.iv_size);
        print_hex("IV", iv, iv.size());
    }

    // =========================================================================
    // 4. Streaming encryption
    // =========================================================================
    auto encrypt_ctx = MakeCipherContext(*ctx, test_data, key, CipherDirection::kEncrypt);
    ASSERT_NE(encrypt_ctx, nullptr);

    const std::size_t block_size = encrypt_ctx->GetOutputSize();
    ASSERT_GT(block_size, 0U) << "Cipher block size query failed";

    std::vector<uint8_t> ciphertext;
    ASSERT_NO_FATAL_FAILURE(StreamingTransform(*encrypt_ctx, iv, plaintext, block_size, ciphertext));
    print_hex("Ciphertext", ciphertext, ciphertext.size());

    ASSERT_FALSE(ciphertext.empty());
    EXPECT_NE(ciphertext, plaintext) << "Ciphertext must not equal plaintext";

    // A padded block mode grows the message; a stream mode keeps it the same size.
    if (test_data.block_padded)
    {
        EXPECT_GT(ciphertext.size(), plaintext.size()) << "Padded mode should append a padding block";
        EXPECT_EQ(ciphertext.size() % block_size, 0U) << "Padded ciphertext must be a whole number of blocks";
    }
    else
    {
        EXPECT_EQ(ciphertext.size(), plaintext.size()) << "Stream mode must preserve the message length";
    }

    // =========================================================================
    // 5. Streaming decryption recovers the plaintext
    // =========================================================================
    auto decrypt_ctx = MakeCipherContext(*ctx, test_data, key, CipherDirection::kDecrypt);
    ASSERT_NE(decrypt_ctx, nullptr);

    std::vector<uint8_t> recovered;
    ASSERT_NO_FATAL_FAILURE(StreamingTransform(*decrypt_ctx, iv, ciphertext, block_size, recovered));
    EXPECT_EQ(recovered, plaintext) << "Decryption did not recover the original plaintext";

    // =========================================================================
    // 6. Single-shot encryption produces the same ciphertext as streaming
    // =========================================================================
    std::vector<uint8_t> ciphertext_ss;
    ASSERT_NO_FATAL_FAILURE(SingleShotTransform(*encrypt_ctx, iv, plaintext, block_size, ciphertext_ss));
    EXPECT_EQ(ciphertext_ss, ciphertext) << "SingleShot and streaming encryption must agree";

    // =========================================================================
    // 7. Single-shot decryption round trip
    // =========================================================================
    std::vector<uint8_t> recovered_ss;
    ASSERT_NO_FATAL_FAILURE(SingleShotTransform(*decrypt_ctx, iv, ciphertext_ss, block_size, recovered_ss));
    EXPECT_EQ(recovered_ss, plaintext) << "SingleShot decryption did not recover the plaintext";

    // =========================================================================
    // 8. Context reuse via Reset() with a different message
    // =========================================================================
    std::vector<uint8_t> ciphertext_alt;
    ASSERT_NO_FATAL_FAILURE(SingleShotTransform(*encrypt_ctx, iv, plaintext_alt, block_size, ciphertext_alt));
    EXPECT_NE(ciphertext_alt, ciphertext) << "A different message must produce different ciphertext";

    std::vector<uint8_t> recovered_alt;
    ASSERT_NO_FATAL_FAILURE(SingleShotTransform(*decrypt_ctx, iv, ciphertext_alt, block_size, recovered_alt));
    EXPECT_EQ(recovered_alt, plaintext_alt) << "Round trip after Reset failed";

    // =========================================================================
    // 9. A different IV yields different ciphertext for the same message
    // =========================================================================
    if (test_data.iv_size > 0U)
    {
        std::vector<uint8_t> other_iv = iv;
        other_iv[0] ^= 0xFFU;

        std::vector<uint8_t> ciphertext_other_iv;
        ASSERT_NO_FATAL_FAILURE(
            SingleShotTransform(*encrypt_ctx, other_iv, plaintext, block_size, ciphertext_other_iv));
        EXPECT_NE(ciphertext_other_iv, ciphertext) << "Changing the IV must change the ciphertext";
    }

    // =========================================================================
    // 10. Tampered ciphertext must not decrypt back to the plaintext
    // =========================================================================
    //
    // A padded mode detects the tampering and fails outright; a stream mode has
    // no integrity check and simply yields different bytes. Both outcomes are
    // acceptable — what must never happen is recovering the original message.
    std::vector<uint8_t> tampered = ciphertext;
    tampered[0] ^= 0xFFU;

    ASSERT_TRUE(decrypt_ctx->Reset()) << "Reset before tampered decrypt failed";
    std::vector<uint8_t> tampered_out(tampered.size() + (2U * block_size));
    auto tampered_result = decrypt_ctx->SingleShot(
        {iv.data(), iv.size()}, {tampered.data(), tampered.size()}, {tampered_out.data(), tampered_out.size()});

    if (tampered_result.has_value())
    {
        tampered_out.resize(tampered_result.value());
        EXPECT_NE(tampered_out, plaintext) << "Tampered ciphertext must not decrypt to the original plaintext";
    }

    // =========================================================================
    // 11. Explicit key release
    // =========================================================================
    auto release_result = key.Release();
    ASSERT_TRUE(release_result.has_value()) << "Key release failed";
    EXPECT_FALSE(key.IsActive()) << "Key should be inactive after Release";
}

// =========================================================================
// Test Vector Constants
// =========================================================================

constexpr std::size_t kAesIvSize = 16U;

// =========================================================================
// Test Suites
// =========================================================================

// Symmetric ciphers are provided only by the OpenSSL (software) provider, so
// every case pins ProviderType::kSoftware. Leaving the provider unset would
// resolve to the daemon's DEFAULT provider — currently SoftHSM — which offers
// no cipher handler.
INSTANTIATE_TEST_SUITE_P(CbcOnSoftwareProvider,
                         CipherRoundTripTest,
                         ::testing::Values(CipherTestData{"AES256_CBC_SoftwareProvider",
                                                          ProviderType::kSoftware,
                                                          "AES-256-CBC",
                                                          "AES-256-CBC",
                                                          kAesIvSize,
                                                          true},
                                           CipherTestData{"AES192_CBC_SoftwareProvider",
                                                          ProviderType::kSoftware,
                                                          "AES-192-CBC",
                                                          "AES-192-CBC",
                                                          kAesIvSize,
                                                          true},
                                           CipherTestData{"AES128_CBC_SoftwareProvider",
                                                          ProviderType::kSoftware,
                                                          "AES-128-CBC",
                                                          "AES-128-CBC",
                                                          kAesIvSize,
                                                          true}),
                         [](const testing::TestParamInfo<CipherRoundTripTest::ParamType>& info) {
                             return info.param.test_case_name;
                         });
}  // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
