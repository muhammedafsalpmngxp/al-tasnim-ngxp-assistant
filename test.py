import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file
load_dotenv()

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")

print("=" * 60)
print("🔍 API KEY CHECK")
print("=" * 60)
print(f"API Key loaded: {'Yes' if api_key else 'No'}")
if api_key:
    print(f"API Key length: {len(api_key)}")
    print(f"API Key prefix: {api_key[:10]}...")
    print(f"API Key suffix: ...{api_key[-10:]}")
else:
    print("❌ API Key not found in .env file")
    print("Make sure GOOGLE_API_KEY is set in your .env file")
    exit(1)

# Configure and test
print("\n🔌 Testing API key...")
genai.configure(api_key=api_key)

try:
    # List models to verify key works
    print("\n📋 Available models:")
    for model in genai.list_models():
        if 'embed' in model.name.lower():
            print(f"  ✅ {model.name} - {model.display_name}")
    print("\n✅ API key is valid!")
    
except Exception as e:
    print(f"\n❌ API key is invalid: {e}")
    print("\n💡 Check your API key in .env file")