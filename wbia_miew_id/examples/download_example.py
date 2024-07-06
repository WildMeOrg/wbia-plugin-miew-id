import requests
import tarfile
import os

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

def extract_tarfile(tar_path, extract_to):
    with tarfile.open(tar_path, "r:gz") as tar:
        top_level_dir = os.path.commonprefix(tar.getnames())
        
        for member in tar.getmembers():
            member_path = os.path.join(extract_to, os.path.relpath(member.name, top_level_dir))
            if member.isdir():
                if not os.path.isdir(member_path):
                    os.makedirs(member_path)
            else:
                if not os.path.isdir(os.path.dirname(member_path)):
                    os.makedirs(os.path.dirname(member_path))
                with open(member_path, 'wb') as f:
                    f.write(tar.extractfile(member).read())

def main():
    url = "https://cthulhu.dyn.wildme.io/public/datasets/beluga_example_miewid.tar.gz"
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    tar_path = os.path.join(script_dir, "beluga_example_miewid.tar.gz")
    extract_to = os.path.join(script_dir, "beluga_example_miewid")

    print(f"Downloading {url} to {tar_path}...")
    download_file(url, tar_path)
    print(f"Downloaded to {tar_path}")

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    print(f"Extracting {tar_path} to {extract_to}...")
    extract_tarfile(tar_path, extract_to)
    print("Extraction completed")

    os.remove(tar_path)
    print(f"Removed {tar_path}")

if __name__ == "__main__":
    main()
