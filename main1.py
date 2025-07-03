from utils.io_utils import single_file, folder_image

def main():
    print("\n=== X-ray Bone Segmentation Processing Menu ===")
    print("1. Process a single X-ray image")
    print("2. Process all X-ray images in a folder")
    while True:
        choice = input("Enter your choice (1 or 2, or 'q' to quit): ").strip().lower()
        if choice in ['1', '2', 'q']:
            break
        print("❌ Invalid choice. Please enter 1, 2, or 'q'.")

    try:
        if choice == '1':
            single_file()
        elif choice == '2':
            folder_image()
        else:
            print("Exiting program.")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
    finally:
        print("\n=== Processing Complete ===")

if __name__ == "__main__":
    main()