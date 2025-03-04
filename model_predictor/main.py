import predict

print("Bienvenido al estimador de precios de bienes raíces.")

town = input("Ingrese la ciudad de interés: ")
max_price = float(input("Ingrese el precio máximo que está dispuesto a pagar: "))

# Obtener predicción
result = predict.predict_price(town, max_price)

# Mostrar resultados
if result:
    for year, price in result.items():
        print(f"Precio estimado en {year}: ${price:.2f}")
