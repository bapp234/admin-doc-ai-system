from src.modules.preprocessing.loader import load_image

def main():
    pages = list(load_image("tests/assets/sample.jpg"))
    print("Pages:", len(pages))
    print("Shape:", pages[0][1].shape)

if __name__ == "__main__":
    main()
