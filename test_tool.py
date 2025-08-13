#!/usr/bin/env python3
"""
Test script to verify that the tools work independently before running the full crew
"""

from tools import heart_predictor, heart_image_display

def test_heart_predictor():
    """Test the heart prediction tool"""
    print("Testing Heart Prediction Tool...")
    try:
        result = heart_predictor._run("Patient has chest pain and shortness of breath")
        print(f"✅ Heart Predictor Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Heart Predictor Error: {e}")
        return False

def test_image_display():
    """Test the image display tool"""
    print("\nTesting Heart Image Display Tool...")
    try:
        result = heart_image_display._run("Anterior Wall MI")
        print(f"✅ Image Display Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Image Display Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING TOOLS INDEPENDENTLY")
    print("=" * 50)
    
    predictor_ok = test_heart_predictor()
    image_ok = test_image_display()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Heart Predictor: {'✅ PASS' if predictor_ok else '❌ FAIL'}")
    print(f"Image Display: {'✅ PASS' if image_ok else '❌ FAIL'}")
    
    if predictor_ok and image_ok:
        print("\n🎉 All tools working! You can now run main.py")
    else:
        print("\n⚠️  Some tools have issues. Check the errors above.")