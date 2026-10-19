#!/usr/bin/env python3
"""A patch of forest in the previewer, scattered from a list of flora kinds, to judge a
change to Data/flora_kinds.txt (see scripts/forest_flora.py) without starting the game.

  scripts/flora_preview.py                                  the game's own list
  scripts/flora_preview.py --module "Native Refurbished"    a module's list (the game's if it has none)
  scripts/flora_preview.py --module "Native Refurbished" --png forest.png
  scripts/flora_preview.py --region steppe_forest --seed 7 --size 50

The choice follows the game (read from its executable, see docs/engine-notes.md): the
region gives a number of groups of trees and of other plants, each group takes one kind
chosen evenly among those of its category that may grow there, and its amount is
density / 1000 x (1 to 2) x (0.40 to 0.45) x 15 x 3 / (groups + 2). Where the game puts
a group's plants is not known: here they are strewn evenly over the square, and the
amount is taken as plants per square metre times --rate (default 0.15, which gives the
game's own forest about a tree every 100 m2). So it shows how two lists compare more than
exactly what a battle will look like; --seed draws other groups, as another battle would.
Grass is left out: it follows the game's grass option. Other options go to build/replay
(--yaw, --pitch, --forward, --frames ...).
"""
import argparse
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forest_flora as ff  # noqa: E402

REGION = {"plain": ff.PLAIN, "steppe": ff.STEPPE, "snow": ff.SNOW, "desert": ff.DESERT,
          "plain_forest": ff.PLAIN_FOREST, "steppe_forest": ff.STEPPE_FOREST,
          "snow_forest": ff.SNOW_FOREST, "desert_forest": ff.DESERT_FOREST}
GROUND = {"plain": "ground_earth", "steppe": "ground_earth", "snow": "ground_earth", "desert": "ground_earth",
          "plain_forest": "ground_forest", "steppe_forest": "ground_forest", "snow_forest": "ground_forest",
          "desert_forest": "ground_earth"}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--module", help="read this module's Data/flora_kinds.txt when it has one")
    ap.add_argument("--region", default="plain_forest", choices=sorted(REGION))
    ap.add_argument("--size", type=float, default=60.0, help="metres across the square (default 60)")
    ap.add_argument("--rate", type=float, default=0.15, help="the game's amount taken as this many plants per square metre (default 0.15)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ground", type=float, default=-2.0, help="where the ground is, metres below the camera (default -2)")
    a, rest = ap.parse_known_args()

    game = os.environ.get("WBM_GAME") or os.path.expanduser("~/Library/Application Support/Steam/steamapps/common/MountBlade Warband")
    source = os.path.join(game, "Data", "flora_kinds.txt")
    if a.module:
        own = os.path.join(game, "Modules", a.module, "Data", "flora_kinds.txt")
        if os.path.exists(own):
            source = own
    kinds, _ = ff.read_kinds(source)
    rng = random.Random(a.seed)
    # Groups of trees and of other plants for each region, as the game draws them.
    groups = {"plain_forest": (rng.randint(2, 5), rng.randint(2, 3)), "steppe_forest": (rng.randint(2, 5), rng.randint(2, 3)),
              "snow_forest": (2, rng.randint(2, 3)), "desert_forest": (rng.randint(3, 4), 2), "desert": (0, 2)}
    tree_groups, other_groups = groups.get(a.region, (rng.randint(3, 6), rng.randint(4, 5)))
    lines = ["@plane %s 0 %g %g 0" % (GROUND[a.region], a.ground, a.size / 2 + 4)]
    counts = {"trees": 0, "other": 0}
    chosen = []
    for category, n in (("trees", tree_groups), ("other", other_groups)):
        allowed = [k for k in kinds if k[1] & REGION[a.region] and bool(k[1] & ff.TREE) == (category == "trees") and not k[1] & ff.GRASS]
        for _ in range(n if allowed else 0):
            name, flags, meshes, body = rng.choice(allowed)
            density = flags >> 32
            amount = density / 1000.0 * (1 + rng.random()) * (0.4 + 0.05 * rng.random()) * 15 * 3 / (n + 2)
            names = [line.split()[0] for line in (body[::2] if flags & ff.TREE else body)]
            many = int(round(amount * a.rate * a.size * a.size))
            chosen.append("%s (%d)" % (name, many))
            for _ in range(many):
                x, z = rng.uniform(-a.size / 2, a.size / 2), rng.uniform(4, 4 + a.size)
                lines.append("%s - %.2f _%g %.2f %d" % (rng.choice(names), x, a.ground, z, rng.randrange(360)))
                counts[category] += 1
    print("groups: " + ", ".join(chosen), file=sys.stderr)
    gallery = os.path.join(root, "build", "flora-preview.txt")
    os.makedirs(os.path.dirname(gallery), exist_ok=True)
    with open(gallery, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("%s, %s: %d trees and %d other plants on %g m square (gallery %s)"
          % (source, a.region, counts["trees"], counts["other"], a.size, gallery), file=sys.stderr)

    stage = os.environ.get("WBM_STAGE") or os.path.join(root, "traces", "20261012-224340", "capture_006472.wbcap")
    env = dict(os.environ)
    if a.module:
        env["WBM_MODULE"] = a.module
    sys.exit(subprocess.call([os.path.join(root, "scripts", "replay.sh"), stage, "--gallery", gallery] + rest, env=env))


if __name__ == "__main__":
    main()
