import asyncio
from bleak import BleakClient
from datetime import datetime
import numpy as np

def decode_data(data):

    has_rr_data = data[0] & 0x10
    heart_rate = data[1]

    rr_data = -1
    if has_rr_data:
        rr_data = int.from_bytes(data[2:4], byteorder='little') / 1024

    if rr_data > 0:
        return heart_rate, rr_data
    return None, None


def calc_hrv(rrs_list):
    diffs = np.diff(rrs_list)
    squared_diffs = diffs ** 2
    rmssd = np.sqrt(np.mean(squared_diffs))
    return np.round(rmssd * 1000, 2)

async def main():
    address = 'D3:8F:46:6B:7D:CF'
    uuid = '00002a37-0000-1000-8000-00805f9b34fb'

    rrs_list = []
    async with BleakClient(address) as client:

        def handle_hr(sender, data):
            now = datetime.now().strftime('%H:%M:%S')
            hr, rrs = decode_data(data)
            if hr is not None and rrs is not None:
                rrs_list.append(rrs)
                if len(rrs_list) > 1:
                    hrv_short = calc_hrv(rrs_list[-60:])
                    hrv_long = calc_hrv(rrs_list[-300:])
                else:
                    hrv_long, hrv_short = -1, -1
                print(now, hr, hrv_long, hrv_short)

        await client.start_notify(uuid, handle_hr)

        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())