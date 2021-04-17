import os
import sys
from glob import glob


def create_list(images_dir, output_file, img_ext=".jpg"):
    ImgList = os.listdir(images_dir)
    
    val_list = []

    for img in ImgList:
        img,ext = img.split(".")
        val_list.append(img)

    with open(os.path.join(images_dir, output_file),'w') as fid:
        for line in val_list[:-1]:
            fid.write(line + "\n")
            
        fid.write(val_list[-1])

def main():
    if len(sys.argv) < 2:
        print("Requires images directory")
        sys.exit(1)
    elif len(sys.argv) < 3:
        images_dir = sys.argv[1]
        output_file = "image_list.txt"
    else:
        images_dir = sys.argv[1]
        output_file = sys.argv[2]

    create_list(images_dir, output_file)
    
if __name__=="__main__":
    main()