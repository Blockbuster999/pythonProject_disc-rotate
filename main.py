import os
import numpy as np
from datetime import datetime
import pandas as pd
from pathlib import Path
from PIL import Image
import pytesseract
import re
from typing import Optional, Tuple
import shutil

# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Jordan\AppData\Local\Programs\Python\Python39\Scripts\pytesseract.exe"

# =================================================
# functions to get timestamp meta data from the ocr of the image
# =================================================
def get_OCR_data(image_path: str) -> str:

    # Open the image
    img = Image.open(image_path)

    # Extract text
    text = pytesseract.image_to_string(img, lang="eng")

    # print(text)
    return text


def extract_datetime_OCR_str(s: str) -> Optional[Tuple[str, str]]:
    """
    Extracts the date and time from a string.

    Returns:
        A tuple (date, time) as strings, e.g., ("2026-01-31", "13:18:00")
        If time in the string has no seconds, adds ':00'.
        Returns None if no date-time pattern is found.
    """

    # Regex pattern: YYYY-MM-DD followed by HH:MM, optionally :SS
    pattern = r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}(?::\d{2})?)'

    match = re.search(pattern, s)
    if match:
        date_part = match.group(1)
        time_part = match.group(2)
        # Add ':00' if seconds are missing
        if len(time_part) == 5:  # HH:MM
            time_part += ":00"
        return date_part, time_part

    return None


# =================================================
# functions to get timestamp meta data from the image file name
# =================================================
def extract_date_time_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Extracts date and time from a filename.

    Expected patterns:
      Date: YYYY-MM-DD
      Time: HH-MM-SS or HH-MM

    Returns:
      (date, time) as ('YYYY-MM-DD', 'HH:MM:SS')
      or None if no valid date/time is found.
    """
    pattern = r'(\d{4}-\d{2}-\d{2})\s+(\d{2}-\d{2}(?:-\d{2})?)'
    match = re.search(pattern, filename)

    if not match:
        return None

    date_part = match.group(1)
    time_part = match.group(2).replace("-", ":")

    # Ensure HH:MM:SS format
    if len(time_part) == 5:  # HH:MM
        time_part += ":00"

    return date_part, time_part


def get_filename_from_path(path: str) -> str:
    """Return the filename (with extension) from a file path."""
    return Path(path).name

# =================================================
# functions to get timestamp meta data from the image exif data
# =================================================
def extract_date_exif(timestamp: str) -> Optional[str]:
    """
    Extract the date from a timestamp string like '2026:01:28 18:13:31'
    and return it as 'yyyy-mm-dd'.

    Args:
        timestamp: string in format 'YYYY:MM:DD HH:MM:SS'

    Returns:
        Date string 'yyyy-mm-dd', or None if parsing fails
    """
    try:
        dt = datetime.strptime(timestamp, "%Y:%m:%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def extract_time_exif(s: str) -> Optional[str]:
    match = re.search(r"(?<!\d)\d{2}:\d{2}:\d{2}(?!\d)", s)
    return match.group(0) if match else None

# =================================================
# functions to process date and time data
# =================================================
def round_time_to_minutes(time_str: str, minutes: int) -> str:
    """
    Round a time (HH:MM:SS) to the nearest `minutes`.

    Args:
        time_str: "HH:MM:SS"
        minutes: rounding interval (e.g. 5, 10, 15, 30)

    Returns:
        Rounded time as "HH:MM:SS"
    """
    dt = datetime.strptime(time_str, "%H:%M:%S")

    # total seconds since midnight
    total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    interval = minutes * 60

    # round to nearest interval
    rounded_seconds = int(round(total_seconds / interval) * interval)

    # wrap around 24h
    rounded_seconds %= 24 * 3600

    hours, remainder = divmod(rounded_seconds, 3600)
    mins, secs = divmod(remainder, 60)

    return f"{hours:02d}:{mins:02d}:{secs:02d}"

# =================================================
# functions to calculate the rotation angle to correct image orientation based on the az/alt object coords
# =================================================
def altaz_to_vector(alt_deg: float, az_deg: float):
    """
    Convert azimuth / altitude to a 3D unit vector.

    az_deg  : azimuth in degrees (0=N, 90=E)
    alt_deg : altitude in degrees (0=horizon, 90=zenith)
    """
    az = np.radians(az_deg)
    alt = np.radians(alt_deg)

    return np.array([
        np.cos(alt) * np.sin(az),  # x (East)
        np.cos(alt) * np.cos(az),  # y (North)
        np.sin(alt)                # z (Up)
    ])


def normalize(v):
    """
        # returns the unit vector
    """
    return v / np.linalg.norm(v)


def signed_angle_between_arcs(A, B, C, degrees=True):
    """
    Signed angle from great-circle arc AB to arc AC at A.
    Positive = right-hand rotation about A.
    """
    A = normalize(np.array(A))
    B = normalize(np.array(B))
    C = normalize(np.array(C))

    # Plane normals
    n1 = normalize(np.cross(A, B))
    n2 = normalize(np.cross(A, C))

    # Unsigned angle
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    # Sign via orientation
    sign = np.sign(np.dot(A, np.cross(n1, n2)))
    angle *= sign

    return np.degrees(angle) if degrees else angle


def wrap_to_180(angle_deg):
    """
    Wrap angle to the range (-180, 180].
    """
    return (angle_deg + 180) % 360 - 180


# =================================================
#  image manipulation functions
# =================================================
def save_image_no_overwrite(img: Image.Image, path: str) -> Path:
    """
    Saves an image without overwriting existing files.
    If the file exists, appends _1, _2, etc.

    Returns the final saved path.
    """
    path = Path(path)
    stem = path.stem
    suffix = path.suffix
    directory = path.parent

    counter = 1
    new_path = path

    while new_path.exists():
        new_path = directory / f"{stem}_{counter}{suffix}"
        counter += 1

    img.save(new_path)
    return new_path


def rotate_image(input_path, output_path, angle_deg):
    """
        input path: path of the image file from the working directory e.g. //input/image.jpg
        output_path: //input/rotated image.jpg
        angle_deg: angle to rotate the image by - positive is CCW
    """
    img = Image.open(input_path)
    rotated = img.rotate(angle_deg, resample=Image.BICUBIC, expand=True)
    save_image_no_overwrite(rotated, output_path)
    print()
    print(f"Rotated image saved to: {output_path}")


def crop_center_1080(img_path: str, save_path: str):
    """
    Crops a 1080x1080 square from the center of the image.

    Args:
        img_path: Path to the original image
        save_path: Path to save the cropped image
    """
    img = Image.open(img_path)
    width, height = img.size

    # Define the size of the crop
    crop_size = 1080

    # Calculate coordinates of the centered crop box
    left = max((width - crop_size) // 2, 0)
    top = max((height - crop_size) // 2, 0)
    right = left + crop_size
    bottom = top + crop_size

    # Crop and save
    cropped_img = img.crop((left, top, right, bottom))
    try:
        # print(f"try saving to {save_path}")
        save_image_no_overwrite(cropped_img, save_path)
    except:
        print(f"error, failed to save cropped image to: {save_path}")
    print(f"Cropped image saved to {save_path}")


def copy_and_rename(src_path, dest_path, new_name_with_ext):
    """
    Copies a file to dest_path and renames it without overwriting existing files.
    """
    src_path = Path(src_path)
    dest_path = Path(dest_path)

    new_name = os.path.splitext(new_name_with_ext)[0]
    suffix = os.path.splitext(new_name_with_ext)[1]

    # Ensure destination directory exists
    dest_path.mkdir(parents=True, exist_ok=True)

    # Get file extension from source
    target = dest_path / f"{new_name}{suffix}"

    counter = 1
    final_target = target

    # Avoid overwriting existing files
    while final_target.exists():
        final_target = dest_path / f"{new_name}_{counter}{suffix}"
        counter += 1

    shutil.copy(src_path, final_target)
    return final_target

# =================================================
# functions to return the alt(E) / az(A) data from the https://www.sunearthtools.com - Annual sun path - tool
# =================================================
def find_columns_for_time(data_path, time_str, sep=None):
    """
    :param sep: separator, default None, function will try to automatically detect the separator
    :param data_path:
    :param time_str: the time to try and find
    :return: the column numbers containing the alt/az data
    """

    df = pd.read_csv(data_path, sep=sep, header=0, engine='python')
    col_headers = df.columns

    e_col, a_col = None, None
    colnum = 0
    for header in col_headers:
        if time_str in header:
            if header.startswith("E"):
                e_col = colnum
            elif header.startswith("A"):
                a_col = colnum
        colnum = colnum + 1
    alt_col = e_col # E = elevation
    az_col = a_col # A = azimuth
    return alt_col,az_col


def find_row_by_first_column(csv_path: str, search_str: str, sep: str = None) -> Optional[int]:
    """
    Search the first column of a CSV for a row matching a specific string.

    Args:
        csv_path: Path to the CSV file
        search_str: String to match in the first column
        sep: CSV separator (default is ';')

    Returns:
        The first matching row as a pandas Series, or None if not found
    """
    # Read the CSV, get pandas data frame
    df = pd.read_csv(csv_path, sep=sep, header=0, engine='python')

    # Get the first column name
    first_col = df.columns[0]

    for row in range(df.shape[0]): # df.shape[0] computes the number of rows in the df
        if df.iloc[row][0] == search_str:
            return row
    return None

# =================================================
# the main functions
# =================================================
def getDateTimeTuple(data_path: str, image_path: str):
    data_name = get_filename_from_path(data_path)
    img_name = get_filename_from_path(image_path)

    OCRdata = get_OCR_data(image_path)

    print(f"Fetching timestamp for: {img_name}")
    print(f"vvvvv - OCR start")
    print(f"{OCRdata}")
    print(f"^^^^^ - OCR end")
    OCRdatetime = extract_datetime_OCR_str(OCRdata)

    if OCRdatetime != None:
        print(OCRdatetime)
        return OCRdatetime
    else:
        print(f"Failed to extract datetime from OCR {img_name}")
        print("attempt to extract from file name instead")
        datetimetuple = extract_date_time_from_filename(img_name)
        if datetimetuple != None:
            print("success, extracted from file name")
            print(datetimetuple)
            return datetimetuple
        else:
            print("failed to extract from file name")
            return None


def get_correction_angle(data_path: str, datetime: Tuple[str,str]):

    if datetime == None:
        return None

    img_date = datetime[0]
    img_time = datetime[1]

    img_time = round_time_to_minutes(img_time, 10)

    # print(f"Time {target_time} → E: {e_value}, A: {a_value}")

    alt_colnum = find_columns_for_time(data_path,img_time)[0]
    az_colnum = find_columns_for_time(data_path,img_time)[1]
    rownum = find_row_by_first_column(data_path, img_date)

    if rownum == None:
        print(f"Data invalid, date not found in {data_path}.")
        return None

    df = pd.read_csv(data_path, sep=None, header=0, engine='python')

    try:
        altdeg = float(df.iloc[rownum, alt_colnum])
        azdeg = float(df.iloc[rownum, az_colnum])
        print()
        print(f"Fetching az-alt data: az {azdeg} , alt {altdeg}")
        # print("Success")
    except:
        print(f"Data invalid: [{rownum}, {alt_colnum}/{az_colnum}] = {df.iloc[rownum, alt_colnum]} , {df.iloc[rownum, az_colnum]}")
        return None

    # A = altaz_to_vector(altdeg, azdeg)  # object
    A = altaz_to_vector(altdeg, azdeg)
    B = altaz_to_vector(90, 0)  # zenith
    C = altaz_to_vector(36.902835, 180)  # celestial south

    # if datetime[0] == "2025-11-08":
    #     print("STOP")

    # to correct the rotation of the image, we must spin the image such that the CN points straight up
    # value indicates CCW angle of displacement from CS to zenith
    angle = signed_angle_between_arcs(A, B, C)

    # angle we need to rotate CCW to align the celestial North Pole to point up
    correction_angle = angle + 180
    # correction_angle = wrap_to_180(correction_angle)
    return correction_angle


def main():

    successes = 0
    fails = 0
    faillist = []

    # Paths
    input_folder = "input"
    rotated_folder = "rotated"
    crop_folder = "cropped"
    renamed_folder = "renamed"
    data_path = "2025-2026 10m chart DS corrected.csv"

    # Make sure folders exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(rotated_folder, exist_ok=True)
    os.makedirs(crop_folder, exist_ok=True)
    os.makedirs(renamed_folder, exist_ok=True)

    # Supported image extensions
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")

    # Process every file in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            extension = os.path.splitext(filename)[1]
            img_path = os.path.join(input_folder, filename)
            img_name = get_filename_from_path(img_path)
            rotated_path = os.path.join(rotated_folder, filename)

            print()
            print()

            datetime = getDateTimeTuple(data_path, img_path)
            if datetime == None:
                fails = fails + 1
                faillist.append(filename)
                continue

            # create a copy of the image, but with datetime as the filename
            renamed_file = datetime[0] + " " + datetime[1].replace(":", "-") + extension

            copy_and_rename(img_path,renamed_folder,renamed_file)

            correction_angle = get_correction_angle(data_path, datetime)
            print()
            print(f"Correction angle = {correction_angle}")

            if correction_angle == None:
                print(f"Failed to get rotate image for: {filename}, due to correction_angle = 'None'")
                fails = fails + 1
                faillist.append(filename)
            else:
                rotated_path = os.path.join(rotated_folder, datetime[0] + " " + datetime[1].replace(":", "-") + extension)
                rotate_image(img_path, rotated_path, correction_angle)

                crop_path = os.path.join(crop_folder, datetime[0] + " " + datetime[1].replace(":", "-") + extension)
                # print(crop_path)
                crop_center_1080(rotated_path, crop_path)

                successes = successes + 1

    print()
    print()
    print("All images rotated!")
    print()
    print(f"Successes: {successes}, Fails: {fails}")

    if fails > 0:
        print()
        print(f"Failed files: {faillist}")
        print("Suggestion, try to rename the failed files to YYYY-MM-DD HH-MM-SS.ext")


if __name__ == '__main__':
    main()
    # rotate_image("1.png", "2.png", 45)
