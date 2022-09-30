import asyncio


async def run(cmd):
    print(f"start to create process {cmd}")
    proc = await asyncio.create_subprocess_shell(cmd)
    print(f'created process for {cmd}', flush=True)
    await proc.wait()


async def gpu_cmd_worker(cmd_queue: asyncio.Queue, gpu_queue: asyncio.Queue):
    while True:
        # Get gpu before get cmd
        gpu_id = await gpu_queue.get()
        cmd = await cmd_queue.get()
        cmd = cmd.format(gpu_id)

        await run(cmd)

        # Notify the queue that the "work item" has been processed.
        gpu_queue.put_nowait(gpu_id)
        cmd_queue.task_done()


async def batch_gpu_cmd(cmdls, gpus=(1, 2, 3), num_proc=2):
    gpu_queue = asyncio.Queue()
    for _ in range(num_proc):
        for i in gpus:
            gpu_queue.put_nowait(i)
    cmd_queue = asyncio.Queue()
    for cmd in cmdls:
        cmd_queue.put_nowait(cmd)

    workers = []
    for i in range(len(gpus) * num_proc):
        worker = asyncio.create_task(gpu_cmd_worker(cmd_queue, gpu_queue))
        workers.append(worker)

    await cmd_queue.join()

    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


