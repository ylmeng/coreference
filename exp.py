import re

def print_labels(corefs):
    prefix = re.compile('\(\d+')
    suffix = re.compile('\d+\)')
    counts = 0
    entity_started = False
    for coref in corefs:
        starts = prefix.findall(coref)
        ends = suffix.findall(coref)
        counts += len(starts)
        counts -= len(ends)
        if starts and not entity_started:
            label = 1  # B
            entity_started = True
            if counts == 0:
                entity_started = False # (0)
        elif ends:
            label = 2
            if counts == 0:
                entity_started = False
        elif counts > 0:
            label = 2
        elif counts == 0:
            label = 0
            entity_started = False
        else:
            raise ValueError("Wrong label")
        print label
