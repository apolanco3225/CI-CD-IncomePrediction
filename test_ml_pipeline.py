
def test_number_of_columns(clean_data):
    num_columns = clean_data.shape[1]
    assert num_columns == 15

def test_column_names(clean_data, column_names):
    
    assert clean_data.columns.tolist() == column_names

def test_cat_column_values(clean_data, cat_features_values_dict):
    for feature, values in cat_features_values_dict.items():
        column_values = clean_data[feature].unique().tolist()
        assert column_values == values



    