# Face Recognition Identifier
Simple face recognition identifier using python open library

## Requirements
- Python 3.8 atau lebih baru.
- Library OpenCV
- Library Flask
- Library `face-recognition`

## How to Run

### Cloning Repository
1. Pada halaman utama repository [GitHub](https://github.com/zshnrg/face-recognition-identifier), buka menu **Clone** lalu salin URL dari repository
2. Buka Terminal
3. Pindah ke direktori yang diinginkan
4. Ketikan `git clone`, lalu tempelkan URL yang telah disalin tadi 
   ```sh
   git clone https://github.com/zshnrg/face-recognition-identifier
   ```
   
5. Pindah ke directory `face-recognition-identifier`
6. Install depedencies yang diperlukan
   ```sh
   pip install -r requirements.txt
   ```

### Running on Server
1. Jalankan server dengan cara
    ```
    flask --app app.py --debug run
    ```
2. Gunakan IP Localhost dan API untuk menjalankan model

### Running on CLI
1. Jalankan program main
    ```
    python main.py
    ```
2. Pilih opsi untuk memulai model
    ```
    1. Register
    2. Identify
    3. Exit
    Select an option: ...
    ```