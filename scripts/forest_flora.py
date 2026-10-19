#!/usr/bin/env python3
"""Denser forests, as data only: writes a module's own Data/flora_kinds.txt.

How the game scatters flora (read from its executable, see docs/engine-notes.md): a
generated scene's region (plain, steppe, ... or one of them as forest) gives a few groups
of each category: trees, other plants (bushes, ferns, rocks), and grass. In a plain or
steppe forest that is 2 to 5 groups of trees and 2 to 3 of other plants. Each group takes
one kind, chosen evenly among the kinds of its category that may grow in the region (the
density plays no part in the choice), and spawns an amount of it in proportion to that
kind's density. So more kinds do not make more plants; more density does.

This copies the game's list, takes the forest flags off every kind that grows in plain or
steppe forest (it still grows in open country as before), and appends after all of the
game's kinds (whose order the game may count on) a forest-only copy of each of them, its
density multiplied: trees by --trees, bushes, ferns and shrubs by --undergrowth, the rest
(rocks, flowers) by 1. The forest then draws on the same kinds as before, at the
multiplied amounts. Snow and desert forests, and grass, are left as they are.

  scripts/forest_flora.py "Native Refurbished"                  # into that module of the game
  scripts/forest_flora.py "Native Refurbished" --trees 2 --undergrowth 2
  scripts/forest_flora.py "Native Refurbished" --remove         # back to the game's own list

A flora_kinds.txt already in the module is kept as flora_kinds.txt.before the first time.
Only scenes generated afterwards change. WBM_GAME names the game's folder, as elsewhere.
"""
import argparse
import os
import sys

PLAIN, STEPPE, SNOW, DESERT = 0x4, 0x8, 0x10, 0x20
PLAIN_FOREST, STEPPE_FOREST, SNOW_FOREST, DESERT_FOREST = 0x400, 0x800, 0x1000, 0x2000
GRASS, TREE, GUARANTEE = 0x80000, 0x400000, 0x1000000
REGIONS = PLAIN | STEPPE | SNOW | DESERT | PLAIN_FOREST | STEPPE_FOREST | SNOW_FOREST | DESERT_FOREST


def read_kinds(path):
    with open(path, "rb") as f:
        lines = f.read().decode("latin-1").splitlines()
    count, at, kinds = int(lines[0]), 1, []
    for _ in range(count):
        name, flags, meshes = lines[at].split()
        flags, meshes = int(flags), int(meshes)
        at += 1
        per_mesh = 2 if flags & TREE else 1  # a tree's mesh has a line of its own after it
        body = lines[at:at + meshes * per_mesh]
        at += meshes * per_mesh
        kinds.append((name, flags, meshes, body))
    return kinds, lines[at:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("module")
    ap.add_argument("--trees", type=float, default=3.0, help="trees in forests, as a multiple of their density (default 3)")
    ap.add_argument("--undergrowth", type=float, default=3.0, help="bushes, ferns and shrubs in forests, as a multiple of their density (default 3)")
    ap.add_argument("--remove", action="store_true", help="take the module's list away again")
    a = ap.parse_args()

    game = os.environ.get("WBM_GAME") or os.path.expanduser("~/Library/Application Support/Steam/steamapps/common/MountBlade Warband")
    source = os.path.join(game, "Data", "flora_kinds.txt")
    folder = os.path.join(game, "Modules", a.module)
    if not os.path.isdir(folder):
        sys.exit("no module called %s in %s" % (a.module, game))
    if os.path.realpath(folder) == os.path.realpath(os.path.join(game, "Modules", "Native")):
        sys.exit("not into Native itself: use a copy of it")
    target = os.path.join(folder, "Data", "flora_kinds.txt")
    before = target + ".before"

    if a.remove:
        if os.path.exists(before):
            os.replace(before, target)
            print("%s: the module's own list as it was" % target)
        elif os.path.exists(target):
            os.remove(target)
            print("%s removed: the module uses the game's list" % target)
        return

    kinds, rest = read_kinds(source)
    added, trees, undergrowth, others = [], 0, 0, 0
    forest = PLAIN_FOREST | STEPPE_FOREST
    for index, (name, flags, meshes, body) in enumerate(kinds):
        forests = flags & forest
        density = flags >> 32
        if not forests or (flags & GRASS and density >= 1000):  # the ground's grass itself follows the grass option
            continue
        if flags & TREE:
            factor = a.trees
            trees += 1
        elif name.startswith(("bush", "big_bush", "fern", "grass_bush", "dry_bush", "thorn")):
            factor = a.undergrowth
            undergrowth += 1
        else:
            factor = 1.0
            others += 1
        kinds[index] = (name, flags & ~forest, meshes, body)  # open country only, as before
        low = (flags & 0xFFFFFFFF & ~REGIONS & ~GUARANTEE) | forests
        more = max(1, min(0xFFFF, int(round(density * factor))))
        added.append(("%s_forest" % name, (more << 32) | low, meshes, body))

    out = [str(len(kinds) + len(added))]
    for name, flags, meshes, body in kinds + added:
        out.append("%s %d %d" % (name, flags, meshes))
        out.extend(body)
    out.extend(rest)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target) and not os.path.exists(before):
        os.replace(target, before)
    with open(target, "wb") as f:
        f.write(("\r\n".join(out) + "\r\n").encode("latin-1"))
    print("%s: the game's %d kinds, %d of them now for open country only, and as many forest-only copies after them:"
          " %d trees (density x%g), %d of undergrowth (x%g), %d others (x1)"
          % (target, len(kinds), len(added), trees, a.trees, undergrowth, a.undergrowth, others))


if __name__ == "__main__":
    main()
