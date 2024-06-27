import os
from minio import Minio
from minio.error import S3Error


def main():
    # Create a MinIO client with the specified endpoint, access key, and secret key.
    minio_client = Minio("localhost:9000",
        access_key="N18PI0XRzRbLKB8il7Uk",
        secret_key="8EdhMimLnfe4mYVVYw3BVWPgP7Z5jVagoz79LqEs",
        secure=False
    )

    bucket_name = "images"

    # Ensure the bucket exists or create it if it doesn't
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created.")
        else:
            print(f"Bucket '{bucket_name}' already exists.")
    except S3Error as e:
        print("Error during bucket handling:", e)
        return

    # File path for the image you want to upload
    file_path = "./frontend/output.png"  # Example: "/home/user/Desktop/photo.jpg"
    file_name = os.path.basename(file_path)  # Extracts the file name from the path

    # Upload the image
    try:
        with open(file_path, "rb") as file_data:
            file_stat = os.stat(file_path)
            minio_client.put_object(
                bucket_name,
                f'predicted/{file_name}',
                file_data,
                file_stat.st_size,
                content_type="image/jpeg"
            )
        print("Upload successful.")
    except S3Error as e:
        print("Error during file upload:", e)
    except FileNotFoundError:
        print("File not found, please check the path.")
    except Exception as e:
        print("An unexpected error occurred:", e)

if __name__ == "__main__":
    main()