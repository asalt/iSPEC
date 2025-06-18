


import faker
import numpy as np 
import pandas as pd




import pandas as pd
from faker import Faker

def generate_fake_data(num_records, column_definitions):
    """
    Generates fake data using Faker and Pandas DataFrames based on provided column 
definitions.

    Args:
        num_records (int): The number of records to generate.
        column_definitions (dict): A dictionary defining the columns.
                                    Keys are column names, values are dictionaries
                                    with 'type' and optional 'options' for the Faker provider.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the generated fake data.
    """

    fake = Faker()
    data = {}  # Dictionary to hold lists of data for each column

    for column_name, column_def in column_definitions.items():
        column_type = column_def['type']
        options = column_def.get('options', {})  # Use .get() to handle missing options

        if column_type == 'name':
            data[column_name] = [fake.name() for _ in range(num_records)]
        elif column_type == 'address':
            data[column_name] = [fake.address() for _ in range(num_records)]
        elif column_type == 'email':
            data[column_name] = [fake.email() for _ in range(num_records)]
        elif column_type == 'phone_number':
            data[column_name] = [fake.phone_number() for _ in range(num_records)]
        elif column_type == 'date':
            data[column_name] = [fake.date_between(start_date='-10y', end_date='today') for _ in range(num_records)]
        elif column_type == 'integer':
            min_value = options.get('min', 0)  # Default min is 0
            max_value = options.get('max', 100)  # Default max is 100
            data[column_name] = [fake.random_int(min=min_value, max=max_value) for _ in range(num_records)]
        elif column_type == 'float':
            l_digits = options.get('l_digits', 2)  # Default 2 decimal places
            r_digits = options.get('r_digits', 2)  # Default 2 decimal places
            data[column_name] = [fake.pyfloat(left_digits=l_digits, right_digits=r_digits, positive=True) for _ in range(num_records)]
        elif column_type == 'text':
            max_nb_chars = options.get('max_nb_chars', 200) #default length is 200
            data[column_name] = [fake.text(max_nb_chars=max_nb_chars) for _ in range(num_records)]
        elif column_type == 'boolean':
            data[column_name] = [fake.boolean() for _ in range(num_records)]
        elif column_type == 'first_name':
            data[column_name] = [fake.first_name() for _ in range(num_records)]
        elif column_type == 'last_name':
            data[column_name] = [fake.last_name() for _ in range(num_records)]
        else:
            print(f"Warning: Unknown column type '{column_type}' for column '{column_name}'. Skipping.")
            continue

    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    # Define your column definitions
    

    column_definitions = {
        'personnel_id': {'type': 'integer', 'options': {'min': 100000, 'max': 900000}},
        'first_name': {'type': 'first_name'},
        'last_name': {'type': 'last_name'},
        'email': {'type': 'email'},
        'phone_number': {'type': 'phone_number'},
        'institution' : {'type' : 'text', 'options': {'max_nb_chars': 20}}
        #'address': {'type': 'address'},
        #'registration_date': {'type': 'date'},
        #'purchase_amount': {'type': 'float', 'options': {'l_digits': 2, 'r_digits': 2}},
        #'peepole_status': {'type': 'boolean'},
        #'description': {'type': 'text', 'options': {'max_nb_chars': 100}}
    }

    # Generate 10 records
    num_records = 10
    df = generate_fake_data(num_records, column_definitions)

    # Print the DataFrame
    print(df)

    # python data_generator.py
    # df.to_csv('fake_customer_data.csv', index=False)

# generate_fake_data()
