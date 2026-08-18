machine = "Pump"
temperature = 85
pressure = 120
rpm = 1500
operating_hours = 2500

print("Machine:", machine)
print("Temperature:", temperature)
print("Pressure:", pressure)
print("RPM:", rpm)
print("Operating hours:", operating_hours)

result = temperature * pressure
print("Temperature × Pressure:", result)

if temperature >= 100:
    print("CRITICAL TEMPERATURE")
elif temperature >= 80:
    print("HIGH TEMPERATURE")
else:
    print("NORMAL TEMPERATURE")

if rpm > 3000:
    print("HIGH RPM")
else:
    print("RPM NORMAL")
