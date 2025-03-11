from controller.data_controller import load_and_prepare_data, engineer_features
from controller.model_controller import train_and_evaluate_model

def main():
    print("Cargando y preparando datos...")
    df_train, df_val, df_test, y_train, y_val, y_test, df_full_train = load_and_prepare_data('data/train.csv')
    
    print("Realizando ingeniería de características...")
    df_train, df_val, df_test = engineer_features(df_train, df_val, df_test, df_full_train)
    
    print("Entrenando y evaluando el modelo...")
    selected_columns = [
        "lotarea", "grlivarea", "1stflrsf", "mssubclass", "overallcond",
        "bsmtunfsf", "garagearea", "yearbuilt", "overallqual", "bsmtfinsf1", "group_neighborhood",
        "group_exterior2nd", "fireplaces", "openporchsf", "2ndflrsf",
        "group_housestyle", "masvnrarea", "lotfrontage", "lotconfig", "yearremodadd", "screenporch",
        'mszoning', 'lotshape', 'landcontour', 'landslope', 'condition1', "bedroomabvgr",
        'roofstyle', 'roofmatl', 'exterior1st', 'exterqual', 'extercond', 'foundation',
        'bsmtqual', 'bsmtexposure', 'heatingqc', 'centralair', 'electrical',
        'functional', 'garagequal', 'paveddrive'
    ]
    X_train = df_train[selected_columns]
    X_val = df_val[selected_columns]
    X_test = df_test[selected_columns]
    
    train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    main()