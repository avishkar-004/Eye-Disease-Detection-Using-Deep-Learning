import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_upload(file):
    if not file or file.filename == '':
        return False, "No file selected"
    if not allowed_file(file.filename):
        return False, "Invalid file type. Allowed: PNG, JPG, JPEG"
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return False, "File too large"
    return True, "Valid"
