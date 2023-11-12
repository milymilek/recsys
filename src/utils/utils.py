def write_scalars(writer, names, scalars, step):
    for name, scalar in zip(names, scalars):
        writer.add_scalar(name, scalar, step)