import easyocr
import string

reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def get_car(license_plate, car_track_ids):
    x1, y1, x2, y2 = license_plate[:4]
    for track in car_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = track
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id

    return -1, -1, -1, -1, -1

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 8:
        return False

    if (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0] in dict_char_to_int.keys()) and \
       (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
       (text[2] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
       (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
       (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()) and \
       (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in dict_char_to_int.keys()):
        return True
    else:
        return False

def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_char_to_int, 1: dict_char_to_int,3: dict_char_to_int, 4: dict_char_to_int, 5: dict_char_to_int, 6: dict_char_to_int, 7: dict_char_to_int,
                2: dict_int_to_char}

    for j in range(len(text)):
        if text[j].isalnum() or text[j] == '.':
            if j in mapping and text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

    return license_plate_

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        # if license_complies_format(text):
        return format_license(text), score
        
    return None, None

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{}\n'.format('car_id', 'car_bbox',
                                            'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                            'license_number_score'))

        for car_id in results.keys():
            if 'car' in results[car_id].keys() and \
                'license_plate' in results[car_id].keys() and \
                'text' in results[car_id]['license_plate'].keys():
                f.write('{},{},{},{},{},{}\n'.format(car_id,
                                                    '[{} {} {} {}]'.format(
                                                        results[car_id]['car']['bbox'][0],
                                                        results[car_id]['car']['bbox'][1],
                                                        results[car_id]['car']['bbox'][2],
                                                        results[car_id]['car']['bbox'][3]),
                                                    '[{} {} {} {}]'.format(
                                                        results[car_id]['license_plate']['bbox'][0],
                                                        results[car_id]['license_plate']['bbox'][1],
                                                        results[car_id]['license_plate']['bbox'][2],
                                                        results[car_id]['license_plate']['bbox'][3]),
                                                    results[car_id]['license_plate']['bbox_score'],
                                                    results[car_id]['license_plate']['text'],
                                                    results[car_id]['license_plate']['text_score']))
        f.close()
