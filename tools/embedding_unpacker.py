
import gzip
import shutil

def __main__():
    gzip_path = '/content/drive/MyDrive/cc.ru.300.bin.gz'
    bin_path = '/content/drive/MyDrive/cc.ru.300.bin'

    with gzip.open(gzip_path, 'rb') as f_in:
        with open(bin_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)