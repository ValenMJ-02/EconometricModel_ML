def display_predictions(predictions):
    """
    Muestra las predicciones en la consola.
    """
    for year, price in predictions.items():
        print(f"Año {year}: ${price:,.2f}")
