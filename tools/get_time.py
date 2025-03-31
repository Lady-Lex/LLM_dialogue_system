"""Tool for retrieving the current date and time."""
from datetime import datetime


class GetTimeRun(object):
    name = "Get Time"
    description = (
        "Useful for when you need to answer questions about the current date or time."
    )

    def run(self, no_use: str) -> str:
        time_now = datetime.now()
        time_now_str = time_now.strftime('%H:%M:%S')
        date_now_str = time_now.strftime('%B %d, %Y')  # e.g., March 31, 2025
        weekday_en = time_now.strftime('%A')  # Monday, Tuesday...

        return f"Today is {weekday_en}, {date_now_str}. The current time is {time_now_str}."

if __name__ == '__main__':
    print(GetTimeRun().run(''))
