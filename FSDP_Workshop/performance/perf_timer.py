# Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

# Summary:
# the utility class Timer is a simple performance timer that is controlled with start/interval/stop and reset.

import time

class Timer:
    """
    timer class with start/interval/stop and reset.

    """
    def __init__(self,):
        self.start_time = None
        self.stop_time = None
        self.latest_time=None
    
    def start(self,):
        self.start_time = time.perf_counter()

    def interval(self,name="interval"):
        if self.start_time is None:
            print(f"timer has not started...")
            return

        current_timing = time.perf_counter() - self.start_time  
        print(f"timing for {name} = {round(current_timing,4)}")

    def stop(self,):
        if self.start_time is None:
            print(f"timer has not started...")
            return
        self.stop_time = time.perf_counter()

        self.latest_time = round(self.stop_time - self.start_time ,6)
        print(f"completion time = {self.latest_time}")

    def reset(self,):
        self.start_time = None
        self.stop_time = None
        self.latest_time = None