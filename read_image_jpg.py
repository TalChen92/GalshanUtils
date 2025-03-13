from PIL import Image

def read_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.show()
            return img
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

if __name__ == "__main__":
    file_path = "temp.jpg"
    img = read_image(file_path)
    a=1