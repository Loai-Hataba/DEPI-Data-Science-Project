import os

def list_directory_contents(directory_path, indent=0):
    try:
        # Check if the directory exists
        if not os.path.exists(directory_path):
            print(f"Error: The directory '{directory_path}' does not exist.")
            return
        
        # Check if the path is a directory
        if not os.path.isdir(directory_path):
            print(f"Error: '{directory_path}' is not a directory.")
            return
        
        if indent == 0:
            print(f"Contents of '{directory_path}':")
            print("-" * 50)
        
        # List all items in the directory
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isdir(item_path):
                print(f"{' ' * indent}Directory: {item}")
                # Recursively list contents of subdirectory
                list_directory_contents(item_path, indent + 4)
            else:
                print(f"{' ' * indent}File: {item}")
            
    except PermissionError:
        print(f"Error: Permission denied to access '{directory_path}'")
    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        directory_path = input("Enter the directory path: ")
    
    list_directory_contents(directory_path)