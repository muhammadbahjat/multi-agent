# from fastapi import FastAPI, BackgroundTasks
# import asyncio
# from datetime import datetime
# import time
# from typing import Dict, List
# from contextlib import asynccontextmanager

# # app = FastAPI()

# class TaskScheduler:
#     def __init__(self):
#         self.tasks: Dict[str, asyncio.Task] = {}
        
#     async def scheduled_task(self, task_id: str, interval: int):
#         while True:
#             current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             print(f"Task {task_id} executed at {current_time}")
#             # Your task logic here
#             await asyncio.sleep(interval)
    
#     def schedule_task(self, task_id: str, interval: int):
#         if task_id not in self.tasks:
#             self.tasks[task_id] = asyncio.create_task(
#                 self.scheduled_task(task_id, interval)
#             )
    
#     def stop_task(self, task_id: str):
#         if task_id in self.tasks:
#             self.tasks[task_id].cancel()
#             del self.tasks[task_id]
    
#     def stop_all_tasks(self):
#         for task in self.tasks.values():
#             task.cancel()
#         self.tasks.clear()

# scheduler = TaskScheduler()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup: Schedule initial tasks
#     scheduler.schedule_task("default_task", 10)  # Run every 60 seconds
#     yield
#     # Shutdown: Clean up tasks
#     scheduler.stop_all_tasks()

# app = FastAPI(lifespan=lifespan)

# @app.post("/schedule/{task_id}")
# async def create_scheduled_task(task_id: str, interval: int = 60):
#     """Schedule a new task to run at specified interval (in seconds)"""
#     scheduler.schedule_task(task_id, interval)
#     return {"message": f"Task {task_id} scheduled to run every {interval} seconds"}

# @app.delete("/schedule/{task_id}")
# async def stop_scheduled_task(task_id: str):
#     """Stop a scheduled task"""
#     scheduler.stop_task(task_id)
#     return {"message": f"Task {task_id} stopped"}

# @app.get("/scheduled-tasks")
# async def list_scheduled_tasks():
#     """List all currently scheduled tasks"""
#     return {"tasks": list(scheduler.tasks.keys())}

# # For one-off background tasks
# @app.post("/one-time-task")
# async def create_one_time_task(background_tasks: BackgroundTasks):
#     """Create a one-time background task"""
#     def background_job():
#         time.sleep(1)  # Simulate work
#         print(f"One-time task executed at {datetime.now()}")
    
#     background_tasks.add_task(background_job)
#     return {"message": "One-time task scheduled"}

# # if __name__ == "__main__":
# #     import uvicorn

from fastapi import FastAPI
import asyncio
from datetime import datetime, time
from typing import Dict, Callable, Optional
from contextlib import asynccontextmanager

class EnhancedTaskScheduler:
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        
    async def scheduled_task(
        self,
        task_id: str,
        interval: int,
        task_func: Callable,
        start_hour: Optional[int] = None,
        end_hour: Optional[int] = None
    ):
        while True:
            current_time = datetime.now()
            current_hour = current_time.hour
            
            # Check if we're within the allowed time window
            if (start_hour is not None and end_hour is not None and 
                start_hour <= current_hour <= end_hour):
                
                try:
                    await task_func()
                except Exception as e:
                    print(f"Error executing task {task_id}: {str(e)}")
                    
            await asyncio.sleep(interval)
    
    def schedule_task(
        self,
        task_id: str,
        interval: int,
        task_func: Callable,
        start_hour: Optional[int] = None,
        end_hour: Optional[int] = None
    ):
        if task_id not in self.tasks:
            self.tasks[task_id] = asyncio.create_task(
                self.scheduled_task(task_id, interval, task_func, start_hour, end_hour)
            )
    
    def stop_task(self, task_id: str):
        if task_id in self.tasks:
            self.tasks[task_id].cancel()
            del self.tasks[task_id]
    
    def stop_all_tasks(self):
        for task in self.tasks.values():
            task.cancel()
        self.tasks.clear()

# Initialize scheduler
scheduler = EnhancedTaskScheduler()

# Define your async task functions
async def async_agent_process_messages(task_id: str):
    print(f"Processing agent messages at {datetime.now()} with task id {task_id}")
    # Your implementation here
    
async def async_process_follow_up_for_client_call_scheduling():
    print(f"Processing follow-up for client call scheduling at {datetime.now()}")
    # Your implementation here

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schedule tasks to run every minute between 1 PM to 9 PM UTC
    scheduler.schedule_task(
        "agent_messages",
        10,  # Every minute
        async_agent_process_messages,
        start_hour=1,
        end_hour=23
    )
    
    scheduler.schedule_task(
        "follow_up_scheduling",
        900,  # Every 15 minutes
        async_process_follow_up_for_client_call_scheduling,
        start_hour=13,
        end_hour=23
    )
    
    yield
    
    # Cleanup on shutdown
    scheduler.stop_all_tasks()

app = FastAPI(lifespan=lifespan)

# API endpoints for managing tasks
@app.get("/tasks")
async def list_tasks():
    """List all scheduled tasks"""
    return {"active_tasks": list(scheduler.tasks.keys())}

@app.post("/tasks/{task_id}/stop")
async def stop_task(task_id: str):
    """Stop a specific task"""
    scheduler.stop_task(task_id)
    return {"message": f"Task {task_id} stopped"}

@app.post("/tasks/{task_id}/start")
async def start_task(
    task_id: str,
    interval: int,
    start_hour: Optional[int] = None,
    end_hour: Optional[int] = None
):
    """Start a specific task"""
    # Map task_id to corresponding function
    task_functions = {
        "agent_messages": async_agent_process_messages,
        "follow_up_scheduling": async_process_follow_up_for_client_call_scheduling
    }
    
    if task_id not in task_functions:
        return {"error": "Invalid task_id"}
        
    scheduler.schedule_task(
        task_id,
        interval,
        task_functions[task_id],
        start_hour,
        end_hour
    )
    return {"message": f"Task {task_id} started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)