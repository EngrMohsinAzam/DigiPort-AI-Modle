"""
Test script to verify all dependencies are installed correctly
Run this script after installing requirements.txt
"""

import sys

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"[OK] {package_name} imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] {package_name} import failed: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow installation and GPU support"""
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow {tf.__version__} imported successfully")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  -> GPU devices found: {len(gpus)}")
            for gpu in gpus:
                print(f"    - {gpu.name}")
        else:
            print("  -> No GPU devices found (CPU mode)")
        
        return True
    except Exception as e:
        print(f"[FAIL] TensorFlow test failed: {e}")
        return False

def test_pytorch():
    """Test PyTorch installation and GPU support"""
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__} imported successfully")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"  -> CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  -> CUDA version: {torch.version.cuda}")
        else:
            print("  -> CUDA not available (CPU mode)")
        
        return True
    except Exception as e:
        print(f"[FAIL] PyTorch test failed: {e}")
        return False

def test_opencv():
    """Test OpenCV installation"""
    try:
        import cv2
        print(f"[OK] OpenCV {cv2.__version__} imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] OpenCV test failed: {e}")
        return False

def test_transformers():
    """Test Hugging Face Transformers"""
    try:
        import transformers
        print(f"[OK] Transformers {transformers.__version__} imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Transformers test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Face Liveness Detection Project Dependencies")
    print("=" * 60)
    print()
    
    results = []
    
    # Core libraries
    print("Core Libraries:")
    print("-" * 60)
    results.append(test_import("numpy", "NumPy"))
    results.append(test_import("pandas", "Pandas"))
    results.append(test_import("PIL", "Pillow"))
    print()
    
    # Deep Learning frameworks
    print("Deep Learning Frameworks:")
    print("-" * 60)
    results.append(test_tensorflow())
    results.append(test_pytorch())
    print()
    
    # Computer Vision
    print("Computer Vision Libraries:")
    print("-" * 60)
    results.append(test_opencv())
    results.append(test_import("skimage", "scikit-image"))
    print()
    
    # ML Libraries
    print("Machine Learning Libraries:")
    print("-" * 60)
    results.append(test_import("sklearn", "scikit-learn"))
    print()
    
    # Transformers
    print("Hugging Face Libraries:")
    print("-" * 60)
    results.append(test_transformers())
    results.append(test_import("datasets", "Datasets"))
    print()
    
    # Additional libraries
    print("Additional Libraries:")
    print("-" * 60)
    results.append(test_import("matplotlib", "Matplotlib"))
    results.append(test_import("mtcnn", "MTCNN"))
    results.append(test_import("gdown", "gdown"))
    print()
    
    # Summary
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Summary: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n[SUCCESS] All dependencies installed successfully!")
        print("You can now run the Jupyter notebooks.")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} dependency/dependencies failed to install.")
        print("Please check the error messages above and install missing packages.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
