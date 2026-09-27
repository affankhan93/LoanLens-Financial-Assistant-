from dotenv import load_dotenv
import bcrypt
import os 
from supabase import create_client


load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_API_KEY")

supabase = create_client(url, key)

# Signup Logic 
def signup(username, password):
    existing = supabase.table("users").select("username").eq("username", username).execute()
    if existing.data:
        return False, "Username already taken"
    
    # The password is converted into hash password
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    supabase.table("users").insert({
        "username" : username,
        "hash_password" : hashed
    }).execute()

    return True, "Account created sucessfully!" 

# Login logic 
def login(username, password):
    response = supabase.table("users").select("*").eq("username", username).execute()

    if not response.data:
        return False, "Username is not found" 

    stored_hash = response.data[0]['hash_password']
    if bcrypt.checkpw(password.encode(), stored_hash.encode()):
        return True, "Login Successful"
    else:
        return False, "Incorrect Password Please Enter Valid Password"

    

