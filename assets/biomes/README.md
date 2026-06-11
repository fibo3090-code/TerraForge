# Biome PBR textures

All textures sourced from [ambientCG.com](https://ambientcg.com/) under the
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) license.

| File pair         | ambientCG asset | Used for              |
|-------------------|-----------------|-----------------------|
| `grass_*.jpg`     | Grass001        | Low elevation, gentle slope |
| `dirt_*.jpg`      | Ground037       | Mid elevation, gentle slope |
| `rock_*.jpg`      | Rock030         | Steep slopes (any elevation) |
| `snow_*.jpg`      | Snow005         | High elevation, gentle slope |

`*_albedo.jpg` = base color, `*_normal.jpg` = OpenGL-convention tangent-space
normal map (Bevy's expected convention).

To refresh or swap a set, download `<asset>_1K-JPG.zip` from ambientCG,
extract the `_Color.jpg` and `_NormalGL.jpg` files, and rename them per the
table above.
