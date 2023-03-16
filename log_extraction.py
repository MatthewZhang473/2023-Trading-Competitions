import re

log_file = 'pyready_trader_go/temp_autotrader.log'
output_file = 'prices_log.csv'

with open(log_file, 'r') as f:
    lines = f.readlines()

data = []

for line in lines:
    match = re.search(r'Time = (\d+\.\d+), WAP = (\d+\.\d+), total_WAP = (\d+\.\d+), best_bid = (\d+), best_ask = (\d+)', line)
    if match:
        time = match.group(1)
        wap = match.group(2)
        total_wap = match.group(3)
        best_bid = match.group(4)
        best_ask = match.group(5)
        data.append((time, wap, total_wap, best_bid, best_ask))

with open(output_file, 'w') as f:
    f.write('Time,WAP,Total_WAP,Best Bid,Best Ask\n')
    for row in data:
        f.write(','.join(row) + '\n')
