import pandas as pd

def determine_trail_club_for_locations(trail_club_file, exact_locations_file, output_filename):
    # Read the trail club and exact locations files into dataframes
    club_df = pd.read_excel(trail_club_file)
    locations_df = pd.read_excel(exact_locations_file)

    # Create a mask for rows with valid Latitude values
    valid_latitude_mask = pd.to_numeric(locations_df['Latitude'], errors='coerce').notnull()

    # Convert the valid Latitude rows to float
    locations_df.loc[valid_latitude_mask, "Latitude"] = locations_df.loc[valid_latitude_mask, "Latitude"].astype(float)

    # Convert the longitude_start and longitude_end columns in club_df to float 
    # (since these represent latitudes as per our previous discussions)
    club_df["longitude_start"] = club_df["longitude_start"].astype(float)
    club_df["longitude_end"] = club_df["longitude_end"].astype(float)

    # Add a new column "Trail club" to the locations dataframe to store the matched club
    locations_df["Trail club"] = ""

    for index, location in locations_df[valid_latitude_mask].iterrows():
        latitude = location["Latitude"]
        assigned = False  # Variable to keep track if the location got assigned to any club

        for _, club in club_df.iterrows():
            lat_min = club["longitude_start"]
            lat_max = club["longitude_end"]

            if lat_min <= latitude <= lat_max:
                locations_df.at[index, "Trail club"] = club["Trail club"]
                print(f"Assigned Trail Club: {club['Trail club']} for Latitude: {latitude}")
                assigned = True
                break

        if not assigned:
            print(f"No Trail Club assigned for Latitude: {latitude}")

    # Save the updated locations dataframe to a new Excel file
    locations_df.to_excel(output_filename, index=False)

if __name__ == "__main__":
    trail_club_file = input("Enter the path to the trail club definition file: ")
    exact_locations_file = input("Enter the path to the exact locations file: ")
    output_filename = input("Enter the path for the output file: ")
    determine_trail_club_for_locations(trail_club_file, exact_locations_file, output_filename)