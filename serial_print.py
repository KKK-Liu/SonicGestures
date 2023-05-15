import time

print('Bluetooth connected...')
print("Start Monitoring...")

time.sleep(1)
cnt = 0
while (True):
    cnt += 1
    print(f"Action_{cnt}: EMPTY")
    time.sleep(0.1)