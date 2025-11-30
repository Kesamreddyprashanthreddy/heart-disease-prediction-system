import requests
import json
import time
from typing import Dict, Any


class APITester:
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self) -> bool:
        print("Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Health check passed: {data}")
                return True
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Health check error: {str(e)}")
            return False
    
    def test_model_info(self) -> bool:
        """Test the model info endpoint."""
        print("\nTesting model info...")
        try:
            response = self.session.get(f"{self.base_url}/model_info")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Model info retrieved:")
                for key, value in data.items():
                    if key != 'features_available':  # Skip long feature list
                        print(f"  {key}: {value}")
                return True
            else:
                print(f"✗ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Model info error: {str(e)}")
            return False
    
    def test_sample_prediction(self) -> bool:
        """Test the sample prediction endpoint."""
        print("\nTesting sample prediction...")
        try:
            response = self.session.get(f"{self.base_url}/predict_sample")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Sample prediction successful:")
                print(f"  Input: {data.get('sample_input', {})}")
                print(f"  Prediction: {data.get('prediction')}")
                print(f"  Interpretation: {data.get('interpretation')}")
                return True
            else:
                print(f"✗ Sample prediction failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"✗ Sample prediction error: {str(e)}")
            return False
    
    def test_single_prediction(self) -> bool:
        """Test single prediction."""
        print("\nTesting single prediction...")
        
        # Test data
        test_data = {
            "data": {
                "age": 30,
                "income": 45000,
                "education_years": 14,
                "experience": 8,
                "city": "Chicago",
                "gender": "Female",
                "married": "No",
                "credit_score": 680,
                "loan_amount": 150000,
                "employment_type": "Full-time"
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Single prediction successful:")
                print(f"  Prediction: {data.get('predictions', [])}")
                if 'single_prediction' in data:
                    print(f"  Interpretation: {data['single_prediction']['interpretation']}")
                return True
            else:
                print(f"✗ Single prediction failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"✗ Single prediction error: {str(e)}")
            return False
    
    def test_batch_prediction(self) -> bool:
        """Test batch prediction."""
        print("\nTesting batch prediction...")
        
        # Test data with multiple records
        test_data = {
            "data": [
                {
                    "age": 25,
                    "income": 35000,
                    "education_years": 12,
                    "experience": 3,
                    "city": "Houston",
                    "gender": "Male",
                    "married": "No",
                    "credit_score": 600,
                    "loan_amount": 100000,
                    "employment_type": "Part-time"
                },
                {
                    "age": 45,
                    "income": 80000,
                    "education_years": 18,
                    "experience": 20,
                    "city": "New York",
                    "gender": "Female",
                    "married": "Yes",
                    "credit_score": 780,
                    "loan_amount": 300000,
                    "employment_type": "Full-time"
                }
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Batch prediction successful:")
                print(f"  Number of predictions: {data.get('prediction_count')}")
                print(f"  Predictions: {data.get('predictions', [])}")
                return True
            else:
                print(f"✗ Batch prediction failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"✗ Batch prediction error: {str(e)}")
            return False
    
    def test_invalid_input(self) -> bool:
        """Test handling of invalid input."""
        print("\nTesting invalid input handling...")
        
        # Test with missing data field
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json={"invalid": "data"},
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 400:
                print(f"✓ Invalid input correctly rejected: {response.json().get('error')}")
                return True
            else:
                print(f"✗ Invalid input not properly handled: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Invalid input test error: {str(e)}")
            return False
    
    def run_all_tests(self) -> None:
        """Run all API tests."""
        print("=" * 60)
        print("ML API TESTING SUITE")
        print("=" * 60)
        
        # Wait for server to be ready
        print("Waiting for API server to start...")
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    print("✓ API server is ready")
                    break
            except:
                if i < max_retries - 1:
                    print(f"  Retrying in 2 seconds... ({i+1}/{max_retries})")
                    time.sleep(2)
                else:
                    print("✗ API server not responding. Make sure it's running.")
                    return
        
        # Run tests
        tests = [
            ("Health Check", self.test_health_check),
            ("Model Info", self.test_model_info),
            ("Sample Prediction", self.test_sample_prediction),
            ("Single Prediction", self.test_single_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Invalid Input", self.test_invalid_input)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                print(f"✗ {test_name} failed with exception: {str(e)}")
        
        # Results summary
        print("\n" + "=" * 60)
        print(f"TEST RESULTS: {passed}/{total} tests passed")
        print("=" * 60)
        
        if passed == total:
            print("🎉 All tests passed! API is working correctly.")
        else:
            print(f"⚠️  {total - passed} test(s) failed. Check the logs above.")


def main():
    """Main function to run API tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the ML API")
    parser.add_argument(
        "--url", 
        default="http://localhost:5000",
        help="Base URL of the API (default: http://localhost:5000)"
    )
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    tester.run_all_tests()


if __name__ == "__main__":
    main()