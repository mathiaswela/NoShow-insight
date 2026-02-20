import pandas as pd
import os

def load_and_prep_data(input_filepath: str, output_filepath: str = None) -> pd.DataFrame:
    print(f"Loading data from: {input_filepath}...")
    df = pd.read_csv(input_filepath)

    print("Changing column names...")
    df = df.rename(columns={
        'Handcap': 'Handicap',
        'Hipertension': 'Hypertension',
        'No-show': 'NoShow'
    })

    print("Handling dates and computing wait times...")
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    df['WaitDays'] = (df['AppointmentDay'].dt.normalize() - df['ScheduledDay'].dt.normalize()).dt.days

    print("Removing data errors (negative age and wait times")
    df = df[df['Age'] > 0]
    df = df[df['WaitDays'] >= 0]

    print("Encoding categorical variables and removing noise...")
    df['NoShow_numeric'] = df['NoShow'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})

    col_drop = [
        'PatientId',
        'AppointmentID',
        'ScheduledDay',
        'AppointmentDay',
        'NoShow',
        'Wait_Category',
        'Neighbourhood'
    ]
    df = df.drop(columns=col_drop, errors='ignore')

    if output_filepath:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        df.to_csv(output_filepath, index=False)
        print(f"Data saved to {output_filepath}")

    return df

if __name__ == "__main__":
    print("Starting data prep pipeline...")

    INPUT_PATH = 'data/noshow_data.csv'
    OUTPUT_PATH = 'data/noshow_cleaned_production.csv'

    processed_df = load_and_prep_data(INPUT_PATH, OUTPUT_PATH)
    print("Data prep pipeline completed successfully!")