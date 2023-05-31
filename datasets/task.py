tasks_voc = {
    "offline":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        },
    "19-1":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            1: [20],
        },
    "19-1b":
        {
            0: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            1: [5],
        },
    "15-5":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            1: [16, 17, 18, 19, 20]
        },
    "15-5s":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            1: [16],
            2: [17],
            3: [18],
            4: [19],
            5: [20]
        },
    "10-5-5":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [11, 12, 13, 14, 15],
            2: [16, 17, 18, 19, 20]
        }
}

datasets = {
    "voc": tasks_voc,
}


def _task_dict(dataset, task, stage):
    assert dataset in datasets and task in datasets[dataset], \
        NotImplementedError()
    task_dict = datasets[dataset][task]
    assert stage in task_dict, f"You should provide a valid stage! [{stage} is out of range]"
    return task_dict


def get_task_labels(dataset, task, stage):
    task_dict = _task_dict(dataset, task, stage)
    new_labels = task_dict[stage]
    old_labels = [label for s in range(stage) for label in task_dict[s]]
    print(f"new_labels: {new_labels}")
    print(f"old_labels: {old_labels}")
    return new_labels, old_labels


def classes_per_task(dataset, task, stage):
    '''
    计算从0到stage下每个task有多少个类别，返回列表
    '''
    task_dict = _task_dict(dataset, task, stage)
    return [len(task_dict[s]) for s in range(stage + 1)]