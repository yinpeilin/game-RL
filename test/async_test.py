import asyncio

async def list_append(test_list, i):
    test_list.append(i)
    pass

async def main(test_list2):
    tasks = [asyncio.create_task(list_append(test_list2, index)) for index in test_list]
    await asyncio.wait(tasks)
    
    return test_list2
if __name__ == '__main__':
    pass

    test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    test_list2 = []
    
    print(asyncio.run(main(test_list2)))
