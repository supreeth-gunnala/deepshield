"""
DeepShield Test Script: Security Engine Test
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.security import AudioSecurityEngine

def main():
    print("=" * 50)
    print("     DEEPSHIELD AUDIO SECURITY ENGINE TEST       ")
    print("=" * 50)

    sample_path = os.path.join(PROJECT_ROOT, "data", "test_samples", "clean_mono_16k.wav")

    # 1. SHA-256 Fingerprint Test
    print("\n[STEP 1] Testing SHA-256 audio hash fingerprinting...")
    sha256_hash = AudioSecurityEngine.compute_sha256(sample_path)
    print(f" -> File: '{sample_path}'")
    print(f" -> SHA-256 Hash: {sha256_hash}")

    assert len(sha256_hash) == 64, "Invalid SHA-256 hash length!"

    # 2. File Security Validation Test
    print("\n[STEP 2] Testing file security and header integrity validation...")
    sec_res = AudioSecurityEngine.validate_file_security(sample_path, filename="clean_mono_16k.wav", max_size_mb=50.0)
    
    print(f" -> Is Valid: {sec_res['is_valid']}")
    print(f" -> File Size: {sec_res['file_size_mb']:.4f} MB")
    print(f" -> Status Message: {sec_res['error_message']}")

    assert sec_res["is_valid"] == True, "Clean sample failed security validation!"

    # 3. Invalid Extension Test
    print("\n[STEP 3] Testing invalid extension rejection...")
    invalid_res = AudioSecurityEngine.validate_file_security(sample_path, filename="malicious_script.exe")
    print(f" -> Invalid File Result: {invalid_res['is_valid']}")
    print(f" -> Rejection Message: {invalid_res['error_message']}")

    assert invalid_res["is_valid"] == False, "Invalid extension should be rejected!"

    print("\n" + "=" * 50)
    print("  AUDIO SECURITY ENGINE TEST PASSED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()
