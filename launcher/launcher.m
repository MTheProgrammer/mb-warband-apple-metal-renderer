// Warband Metal launcher: a window in which the renderer's options are chosen,
// and a button that starts Mount & Blade: Warband with the renderer library
// from this app bundle loaded into it. No game file is modified.
//
// Every option is one of the library's WBM_* environment variables (see
// README.md); the window is another way of setting them. A variable already
// set in the environment the launcher is started from wins over the window, so
//   WBM_AO=0 "Warband Metal.app/Contents/MacOS/warband-metal"
// still does what it says. The choices are kept in
// ~/Library/Application Support/warband-metal/settings.plist.
//
// For checking it without the game:
//   warband-metal --print            what would be set, and exit
//   warband-metal --snapshot out.png a picture of the window, and exit

#import <AppKit/AppKit.h>
#include <unistd.h>

typedef NS_ENUM(int, Kind) { kToggle, kSlider, kChoice };

// An option. `everything`, `fast` and `plain` are its values in the three
// presets; an empty value leaves the variable unset. For a choice, `choices`
// is "value=label|value=label"; for a toggle, on is `everything` (or "1").
// The second preset in the list is made from the first (see preset_value).
typedef struct {
    const char *group, *env, *title, *help;
    Kind kind;
    const char *everything, *fast, *plain;
    const char *choices;
    double low, high, step;
} Option;

static const Option k_options[] = {
    // The game's own settings, as its launcher (MBWLauncher.app) offers them: the name is "_" and the key in rgl_config.txt
    // (see game_value), the first value is the game's default, and the presets leave them alone. Its anti-aliasing is left
    // out: under Metal the multisampling below takes its place.
    {"Video", "_display", "Resolution", "The size the game draws at. Below the screen's own, the picture is enlarged to fill the screen.", kChoice, "", "", "", NULL, 0, 0, 0},
    {"Video", "_start_windowed", "Start in a window", "Otherwise the game fills the screen.", kToggle, "0", "", "", NULL, 0, 0, 0},
    {"Video", "_force_vsync", "Vertical sync", "Each frame waits for the display, so the picture does not tear.", kToggle, "1", "", "", NULL, 0, 0, 0},
    {"Video", "_max_framerate", "Frame rate limit", "Fewer frames means a cooler, quieter machine.", kChoice, "60", "", "",
     "30=30|60=60|90=90|120=120|144=144|240=240|1000=None", 0, 0, 0},
    {"Video", "_texture_detail", "Texture detail", "Lower loads smaller textures, for graphics cards with little memory.", kSlider, "100", "", "", NULL, 0, 100, 5},
    {"Video", "_shadowmap_quality", "Shadow detail", "The resolution of the game's shadow map of the scenery.", kChoice, "2", "", "",
     "0=Low|1=Medium|2=High|3=Ultra high", 0, 0, 0},

    {"Shadows and light", "WBM_SHADOWS", "Shadow edges", "How the edges of the game's shadows are filtered.", kChoice, "2", "1", "0",
     "0=As the game has them|1=Soft|2=Soft, wider with distance", 0, 0, 0},
    {"Shadows and light", "WBM_DYNAMIC_SHADOWS", "Shadows from characters", "People, horses and gear cast shadows on everything, hair included.", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"Shadows and light", "WBM_AO", "Ambient occlusion", "Corners and creases are darkened a little.", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"Shadows and light", "WBM_CONTACT_SHADOWS", "Contact shadows", "Short shadows where things touch, finer than the shadow map.", kToggle, "1", "0", "0", NULL, 0, 0, 0},
    {"Shadows and light", "WBM_SKY_AMBIENT", "Light from the sky and the ground", "Shade is tinted by the sky and filled by light off the ground.", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"Shadows and light", "WBM_BOUNCE_LIGHT", "Bounce light", "Lit surfaces throw a little of their light and colour on what is near. 0 for none.", kSlider, "1", "0", "0", NULL, 0, 3, 0.25},
    {"Shadows and light", "WBM_TORCH_SHADOWS", "Shadows from torches", "Torches, candles and fires cast shadows, on everything in the room. 0 for none.", kSlider, "0.8", "0", "0", NULL, 0, 1, 0.1},
    {"Shadows and light", "WBM_TORCH_LIGHTING", "Torchlight per pixel", "Things lit by torches show their relief and shine in that light. Needs the shadows from torches.", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"Shadows and light", "WBM_ROOM_PROBE", "Rooms light themselves", "Indoors, metal reflects the room and shade takes its colour, not the sky's.", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"Shadows and light", "WBM_LIGHT_SHAFTS", "Light shafts", "Rays from the sun when it is in view.", kToggle, "1", "1", "0", NULL, 0, 0, 0},

    {"Materials and colour", "WBM_PBR", "Highlights", "How surfaces shine.", kChoice, "3", "3", "0",
     "0=As the game has them|1=Microfacet|2=With metals|3=With metals, reflecting the sky", 0, 0, 0},
    {"Materials and colour", "WBM_MATERIALS", "Materials by texture", "Metal, cloth, leather, skin, wood and stone each shine their own way.", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"Materials and colour", "WBM_SRGB", "Exact texture colours", "Textures are decoded by the graphics card, before filtering.", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"Materials and colour", "WBM_TRILINEAR", "Smooth texture detail", "Textures blend from sharp to coarse with distance instead of switching at a visible line.", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"Materials and colour", "WBM_TONEMAP", "Filmic tone map", "Bright areas roll off smoothly instead of clipping.", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"Materials and colour", "WBM_HDR", "HDR output", "Highlights go above white on displays that can show it.", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"Materials and colour", "WBM_PARALLAX", "Depth in stone", "Stones stand out from their joints on walls and buildings nearby, by this many metres. 0 for none.", kSlider, "0.05", "0", "0", NULL, 0, 0.12, 0.01},
    {"Materials and colour", "WBM_WATER", "Water", "Glitter from the sun, paler shallows and foam at the shore.", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"Materials and colour", "WBM_WATER_REFLECTIONS", "Reflections in water", "Water mirrors banks, trees and people, and bends the view of its bed. Costs most where water fills the screen.", kToggle, "1", "0", "0", NULL, 0, 0, 0},

    {"People", "WBM_SSS", "Skin", "Light scatters in skin: soft at the edge of a shadow, glowing through ears.", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"People", "WBM_EYES", "Eyes", "A highlight and a reflection in the eyes, which also move a little.", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"People", "WBM_PEOPLE_HAIR", "Hair strands", "Loose strands over the hair and beards of people nearby.", kToggle, "1", "0", "0", NULL, 0, 0, 0},

    // Horses' coat, fur layers and mane and tail strands (WBM_FUR, WBM_FUR_SHELLS, WBM_HAIR_STRANDS) need a texture pack's
    // coat and hair maps, which the release does not bring: they are left out of the window.

    {"Trees and grass", "WBM_WIND", "Wind", "How much trees, bushes and grass sway. 0 for none.", kSlider, "1", "1", "0", NULL, 0, 2, 0.1},
    {"Trees and grass", "WBM_FOLIAGE", "Light through leaves", "Sunlight coming through leaves seen from their far side.", kSlider, "0.6", "0.6", "0", NULL, 0, 1, 0.1},

    {"Picture", "WBM_MSAA", "Multisampling", "Smooths edges as they are drawn. The costliest setting.", kChoice, "1", "1", "4", "1=Off|2=2x|4=4x", 0, 0, 0},
    {"Picture", "WBM_FXAA", "Edge smoothing", "Smooths edges after the scene is drawn. Cheap.", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"Picture", "WBM_ALPHA_HASH", "Soft edges on leaves and hair", "Cut-out edges are smoothed over several frames. Only with temporal enlarging (Speed).", kToggle, "1", "1", "0", NULL, 0, 0, 0},
    {"Picture", "WBM_MOTION_BLUR", "Motion blur", "Blur from the camera's movement, and of things that move: swords, legs, horses.", kToggle, "0", "0", "0", NULL, 0, 0, 0},
    {"Picture", "WBM_SHARPEN", "Sharpening", "0 for none. About 0.3 is plenty together with upscaling.", kSlider, "0", "0.3", "0", NULL, 0, 1, 0.05},

    {"Picture", "WBM_CONTRAST", "Contrast", "Deeper darks and brighter lights, or flatter. 0 leaves the picture as it is.", kSlider, "0", "0", "0", NULL, -1, 1, 0.05},
    {"Picture", "WBM_SATURATION", "Saturation", "Stronger or weaker colours. 0 leaves them as they are.", kSlider, "0", "0", "0", NULL, -1, 1, 0.05},
    {"Picture", "WBM_WARMTH", "Warmth", "Towards warm (right) or cool (left). 0 leaves it as it is.", kSlider, "0", "0", "0", NULL, -1, 1, 0.05},

    {"Speed", "WBM_UPSCALE", "Draw smaller and enlarge", "The scene is drawn smaller and enlarged by MetalFX; the interface stays sharp.", kChoice, "", "0.67", "",
     "=Off|0.8=80% of the size|0.67=67% of the size|0.5=Half the size", 0, 0, 0},
    {"Speed", "WBM_UPSCALE_MODE", "Enlarge with", "Temporal builds the picture from several frames: nearer the full-size picture, and smooths edges itself, for a few milliseconds more.", kChoice, "", "", "",
     "=Spatial|temporal=Temporal", 0, 0, 0},
    {"Speed", "WBM_MENU_FPS", "Frame rate in menus", "A lower limit for menus and loading screens than the one under Video.", kChoice, "", "60", "", "=The game's own|60=60|30=30", 0, 0, 0},

    {"Game", "_language", "Language", "The game's text.", kChoice, "en", "", "", NULL, 0, 0, 0},
    {"Game", "_enable_blood", "Blood", "Blood on hits and on the ground.", kToggle, "1", "", "", NULL, 0, 0, 0},
    {"Game", "_cheat_mode", "Cheats", "The game's cheat keys, such as Ctrl+T to see the whole map.", kToggle, "0", "", "", NULL, 0, 0, 0},
    {"Game", "_enable_version_check", "Version check", "The game looks for a newer version when it starts.", kToggle, "1", "", "", NULL, 0, 0, 0},
    {"Game", "_enable_edit_mode", "Edit mode", "The game's own scene editor and free camera: Ctrl+E in a scene.", kToggle, "0", "", "", NULL, 0, 0, 0},

    {"Sound", "_disable_sound", "Sound", "All of the game's sounds.", kToggle, "1", "", "", NULL, 0, 0, 0},
    {"Sound", "_disable_music", "Music", "The game's music.", kToggle, "1", "", "", NULL, 0, 0, 0},
    {"Sound", "_disable_frequency_variation", "Pitch variation", "Repeated sounds vary a little in pitch.", kToggle, "1", "", "", NULL, 0, 0, 0},
};
static const int k_option_count = sizeof k_options / sizeof k_options[0];
static NSString *const k_presets[] = {@"Everything on", @"Everything but hair", @"Faster", @"As the game looks"};
static const int k_preset_count = 4;

static NSString *S(const char *s) { return s ? @(s) : @""; }

static NSString *support_dir(void) { return [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support/warband-metal"]; }
static NSString *settings_path(void) { return [support_dir() stringByAppendingPathComponent:@"settings.plist"]; }
static NSString *saved_game_path(void) { return [support_dir() stringByAppendingPathComponent:@"game-path"]; }
static NSString *module_file(void) { return [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support/MBWarband/last_module_warband"]; }
static NSString *binary_in(NSString *game) { return [game stringByAppendingPathComponent:@"Mount and Blade.app/Contents/MacOS/Mount and Blade"]; }
static BOOL is_game(NSString *game) { return game.length && [[NSFileManager defaultManager] isExecutableFileAtPath:binary_in(game)]; }

// "Everything but hair" is "Everything on" without people's hair strands, which cost the most of all.
static BOOL is_game_setting(const Option *o) { return o->env[0] == '_'; }
static NSString *game_folder(void) { return [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support/MBWarband"]; }
static NSString *game_config(void) { return [game_folder() stringByAppendingPathComponent:@"rgl_config.txt"]; }
static NSString *language_file(void) { return [game_folder() stringByAppendingPathComponent:@"language.txt"]; }
static NSString *trimmed(NSString *s) { return [s stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet]; }

// The game's rgl_config.txt, "key = value" a line, as a dictionary (empty when the game has not been run yet).
static NSDictionary<NSString *, NSString *> *read_config(void)
{
    NSMutableDictionary *config = [NSMutableDictionary dictionary];
    NSString *text = [NSString stringWithContentsOfFile:game_config() encoding:NSUTF8StringEncoding error:nil];
    for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
        NSArray<NSString *> *parts = [line componentsSeparatedByString:@"="];
        if (parts.count == 2 && trimmed(parts[0]).length) config[trimmed(parts[0])] = trimmed(parts[1]);
    }
    return config;
}

// Rewrites those lines of rgl_config.txt in place, and adds the ones it lacks.
static void write_config(NSDictionary<NSString *, NSString *> *changes)
{
    NSString *text = [NSString stringWithContentsOfFile:game_config() encoding:NSUTF8StringEncoding error:nil] ?: @"";
    NSMutableArray<NSString *> *lines = [[text componentsSeparatedByString:@"\n"] mutableCopy];
    NSMutableSet<NSString *> *left = [NSMutableSet setWithArray:changes.allKeys];
    BOOL changed = NO;
    for (NSUInteger i = 0; i < lines.count; i++) {
        NSArray<NSString *> *parts = [lines[i] componentsSeparatedByString:@"="];
        NSString *key = parts.count == 2 ? trimmed(parts[0]) : nil, *value = key ? changes[key] : nil;
        if (!value) continue;
        [left removeObject:key];
        if ([trimmed(parts[1]) isEqualToString:value]) continue;
        lines[i] = [NSString stringWithFormat:@"%@ = %@", key, value];
        changed = YES;
    }
    for (NSString *key in [left.allObjects sortedArrayUsingSelector:@selector(compare:)]) { // a blank line after each, as the game writes it
        NSUInteger at = lines.count && !lines.lastObject.length ? lines.count - 1 : lines.count;
        [lines insertObjects:@[[NSString stringWithFormat:@"%@ = %@", key, changes[key]], @""] atIndexes:[NSIndexSet indexSetWithIndexesInRange:NSMakeRange(at, 2)]];
        changed = YES;
    }
    if (!changed) return;
    [[NSFileManager defaultManager] createDirectoryAtPath:game_folder() withIntermediateDirectories:YES attributes:nil error:nil];
    [[lines componentsJoinedByString:@"\n"] writeToFile:game_config() atomically:YES encoding:NSUTF8StringEncoding error:nil];
}

// A game setting as the window shows it, read from the game's files; nil when they do not say. "_display" is
// display_width and display_height together, "_language" is language.txt, and a "disable_" key is shown the other
// way round ("Music" on is disable_music = 0).
static NSString *game_value(const Option *o, NSDictionary<NSString *, NSString *> *config)
{
    NSString *key = [S(o->env) substringFromIndex:1];
    if ([key isEqualToString:@"display"])
        return config[@"display_width"] && config[@"display_height"] ? [NSString stringWithFormat:@"%@x%@", config[@"display_width"], config[@"display_height"]] : nil;
    if ([key isEqualToString:@"language"]) {
        NSString *code = trimmed([NSString stringWithContentsOfFile:language_file() encoding:NSUTF8StringEncoding error:nil] ?: @"");
        return code.length ? code : nil;
    }
    NSString *value = config[key];
    if (value && [key hasPrefix:@"disable_"]) return value.intValue ? @"0" : @"1";
    return value;
}

// The game's files made to say what the window does: the reverse of game_value.
static void write_game_settings(NSDictionary<NSString *, NSString *> *values)
{
    NSMutableDictionary *changes = [NSMutableDictionary dictionary];
    for (int i = 0; i < k_option_count; i++) {
        const Option *o = &k_options[i];
        NSString *key = [S(o->env) substringFromIndex:1], *value = values[S(o->env)];
        if (!is_game_setting(o) || !value.length) continue;
        if ([key isEqualToString:@"display"]) {
            NSArray<NSString *> *size = [value componentsSeparatedByString:@"x"];
            if (size.count == 2) changes[@"display_width"] = size[0], changes[@"display_height"] = size[1];
        } else if ([key isEqualToString:@"language"]) {
            if (![game_value(o, @{}) isEqualToString:value]) [value writeToFile:language_file() atomically:YES encoding:NSUTF8StringEncoding error:nil];
        } else if ([key hasPrefix:@"disable_"])
            changes[key] = value.intValue ? @"0" : @"1";
        else
            changes[key] = value;
    }
    write_config(changes);
}

// The sizes offered for the game: the main screen's own (in points, as the game counts them), the sizes the screen
// can be set to and the four the game's launcher lists, all no larger than the screen.
static NSString *display_choices(void)
{
    NSSize screen = NSScreen.mainScreen.frame.size;
    int sw = (int)screen.width, sh = (int)screen.height;
    NSMutableOrderedSet<NSValue *> *sizes = [NSMutableOrderedSet orderedSetWithObject:[NSValue valueWithSize:screen]];
    NSArray *modes = CFBridgingRelease(CGDisplayCopyAllDisplayModes(CGMainDisplayID(), NULL));
    for (id mode in modes)
        [sizes addObject:[NSValue valueWithSize:NSMakeSize(CGDisplayModeGetWidth((__bridge CGDisplayModeRef)mode), CGDisplayModeGetHeight((__bridge CGDisplayModeRef)mode))]];
    for (NSString *s in @[@"1024x768", @"1280x720", @"1600x900", @"1920x1080"]) {
        NSArray<NSString *> *wh = [s componentsSeparatedByString:@"x"];
        [sizes addObject:[NSValue valueWithSize:NSMakeSize(wh[0].intValue, wh[1].intValue)]];
    }
    NSArray<NSValue *> *sorted = [sizes.array sortedArrayUsingComparator:^NSComparisonResult(NSValue *a, NSValue *b) {
      NSSize x = a.sizeValue, y = b.sizeValue;
      return x.width != y.width ? (x.width > y.width ? NSOrderedAscending : NSOrderedDescending) : x.height > y.height ? NSOrderedAscending : x.height < y.height ? NSOrderedDescending : NSOrderedSame;
    }];
    NSMutableArray<NSString *> *choices = [NSMutableArray array];
    for (NSValue *v in sorted) {
        int w = (int)v.sizeValue.width, h = (int)v.sizeValue.height;
        if (w < 1024 || w > sw || h > sh) continue;
        [choices addObject:[NSString stringWithFormat:@"%dx%d=%d × %d%@", w, h, w, h, w == sw && h == sh ? @", the screen's" : @""]];
    }
    return [choices componentsJoinedByString:@"|"];
}

// The game's languages: the folders of its "languages" folder, English among them.
static NSString *language_choices(NSString *game)
{
    NSDictionary *names = @{@"cns" : @"Chinese (simplified)", @"cnt" : @"Chinese (traditional)", @"cz" : @"Čeština", @"da" : @"Dansk", @"de" : @"Deutsch",
                            @"el" : @"Ελληνικά", @"en" : @"English", @"es" : @"Español", @"fi" : @"Suomi", @"fr" : @"Français", @"hr" : @"Hrvatski",
                            @"hu" : @"Magyar", @"it" : @"Italiano", @"ja" : @"日本語", @"no" : @"Norsk", @"nl" : @"Nederlands", @"pl" : @"Polski",
                            @"pt" : @"Português", @"ru" : @"Русский", @"sk" : @"Slovenčina", @"sl" : @"Slovenščina", @"sv" : @"Svenska", @"tr" : @"Türkçe"};
    NSMutableOrderedSet<NSString *> *codes = [NSMutableOrderedSet orderedSetWithObject:@"en"];
    NSString *folder = [game stringByAppendingPathComponent:@"languages"];
    for (NSString *name in [[NSFileManager defaultManager] contentsOfDirectoryAtPath:folder error:nil]) {
        BOOL dir = NO;
        if (![name hasPrefix:@"."] && [[NSFileManager defaultManager] fileExistsAtPath:[folder stringByAppendingPathComponent:name] isDirectory:&dir] && dir) [codes addObject:name];
    }
    NSArray<NSString *> *sorted = [codes.array sortedArrayUsingComparator:^NSComparisonResult(NSString *a, NSString *b) {
      return [(names[a] ?: a) localizedCaseInsensitiveCompare:(names[b] ?: b)];
    }];
    NSMutableArray<NSString *> *choices = [NSMutableArray array];
    for (NSString *code in sorted) [choices addObject:[NSString stringWithFormat:@"%@=%@", code, names[code] ?: code]];
    return [choices componentsJoinedByString:@"|"];
}

static NSString *preset_value(const Option *o, int preset)
{
    if (preset == 1) {
        if (!strcmp(o->env, "WBM_PEOPLE_HAIR")) return @"0";
        return S(o->everything);
    }
    return S(preset == 0 ? o->everything : preset == 2 ? o->fast : o->plain);
}

// WBM_GAME, then the Steam default, then the folder chosen on an earlier run.
static NSString *find_game(void)
{
    NSString *saved = [[NSString stringWithContentsOfFile:saved_game_path() encoding:NSUTF8StringEncoding error:nil] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
    NSArray<NSString *> *candidates = @[S(getenv("WBM_GAME")), [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support/Steam/steamapps/common/MountBlade Warband"], saved ?: @""];
    for (NSString *c in candidates)
        if (is_game(c)) return c;
    return nil;
}

@interface Launcher : NSObject <NSApplicationDelegate>
@property NSWindow *window;
@property NSMutableDictionary<NSString *, NSString *> *values; // by variable name; also "_skip"
@property NSMutableDictionary<NSString *, NSControl *> *controls;
@property NSMutableDictionary<NSString *, NSTextField *> *readouts;
@property NSPopUpButton *presets, *modules;
@property NSTextField *gameLabel;
@property NSButton *skip;
@property NSString *game;
@end

@implementation Launcher

- (void)load
{
    self.values = [NSMutableDictionary dictionary];
    for (int i = 0; i < k_option_count; i++) self.values[S(k_options[i].env)] = preset_value(&k_options[i], 0);
    NSDictionary *saved = [NSDictionary dictionaryWithContentsOfFile:settings_path()];
    for (NSString *key in saved)
        if ([saved[key] isKindOfClass:NSString.class]) self.values[key] = saved[key];
    NSDictionary<NSString *, NSString *> *config = read_config(); // the game's own settings: as the game has them now
    for (int i = 0; i < k_option_count; i++)
        if (is_game_setting(&k_options[i])) {
            NSString *value = game_value(&k_options[i], config);
            if (!value && !strcmp(k_options[i].env, "_display")) value = [NSString stringWithFormat:@"%.0fx%.0f", NSScreen.mainScreen.frame.size.width, NSScreen.mainScreen.frame.size.height];
            if (value) self.values[S(k_options[i].env)] = value;
        }
}

- (void)save
{
    [[NSFileManager defaultManager] createDirectoryAtPath:support_dir() withIntermediateDirectories:YES attributes:nil error:nil];
    [self.values writeToFile:settings_path() atomically:YES];
}

// What the game is started with: the window's choices, under whatever the environment already says.
- (NSDictionary<NSString *, NSString *> *)environment
{
    NSMutableDictionary *env = [NSMutableDictionary dictionary];
    for (int i = 0; i < k_option_count; i++) {
        NSString *name = S(k_options[i].env), *value = self.values[name];
        if (is_game_setting(&k_options[i]) || getenv(k_options[i].env) || !value.length) continue;
        env[name] = value;
    }
    return env;
}

- (int)matchingPreset
{
    for (int p = 0; p < k_preset_count; p++) {
        BOOL same = YES;
        for (int i = 0; i < k_option_count && same; i++) {
            if (is_game_setting(&k_options[i])) continue;
            NSString *have = self.values[S(k_options[i].env)], *want = preset_value(&k_options[i], p);
            same = k_options[i].kind == kSlider ? fabs(have.doubleValue - want.doubleValue) < 1e-6 : [have isEqualToString:want];
        }
        if (same) return p;
    }
    return -1;
}

- (NSString *)show:(double)v step:(double)step { return step >= 1 ? [NSString stringWithFormat:@"%.0f", v] : [NSString stringWithFormat:@"%.3g", v]; }

- (void)refresh
{
    for (int i = 0; i < k_option_count; i++) {
        const Option *o = &k_options[i];
        NSString *name = S(o->env), *value = self.values[name];
        NSControl *control = self.controls[name];
        if (o->kind == kToggle) ((NSButton *)control).state = value.doubleValue != 0 ? NSControlStateValueOn : NSControlStateValueOff;
        else if (o->kind == kSlider) {
            ((NSSlider *)control).doubleValue = value.doubleValue;
            self.readouts[name].stringValue = [self show:value.doubleValue step:o->step];
        } else {
            NSPopUpButton *pop = (NSPopUpButton *)control;
            NSInteger at = [pop indexOfItemWithRepresentedObject:value];
            if (at < 0 && is_game_setting(o) && value.length) { // a value of the game's that the list lacks, such as a size set by hand
                [pop addItemWithTitle:[value stringByReplacingOccurrencesOfString:@"x" withString:@" × "]];
                pop.lastItem.representedObject = value;
                at = pop.numberOfItems - 1;
            }
            [pop selectItemAtIndex:at < 0 ? 0 : at];
        }
        control.enabled = getenv(o->env) == NULL; // set from outside: not the window's to change
    }
    int preset = [self matchingPreset];
    [self.presets selectItemAtIndex:preset < 0 ? k_preset_count : preset];
    self.gameLabel.stringValue = self.game ? [self.game stringByAbbreviatingWithTildeInPath] : @"Not found: choose the \"MountBlade Warband\" folder";
    self.skip.state = [self.values[@"_skip"] isEqualToString:@"1"] ? NSControlStateValueOn : NSControlStateValueOff;
}

- (void)changed:(NSControl *)sender
{
    const Option *o = &k_options[sender.tag];
    NSString *value;
    if (o->kind == kToggle) value = ((NSButton *)sender).state == NSControlStateValueOn ? (S(o->everything).doubleValue != 0 ? S(o->everything) : @"1") : @"0";
    else if (o->kind == kSlider) value = [self show:round(((NSSlider *)sender).doubleValue / o->step) * o->step step:o->step];
    else value = ((NSPopUpButton *)sender).selectedItem.representedObject;
    self.values[S(o->env)] = value;
    [self refresh];
}

- (void)presetChosen:(NSPopUpButton *)sender
{
    NSInteger p = sender.indexOfSelectedItem;
    if (p >= 0 && p < k_preset_count)
        for (int i = 0; i < k_option_count; i++)
            if (!is_game_setting(&k_options[i])) self.values[S(k_options[i].env)] = preset_value(&k_options[i], (int)p);
    [self refresh];
}

- (void)skipChanged:(NSButton *)sender { self.values[@"_skip"] = sender.state == NSControlStateValueOn ? @"1" : @"0"; }

// A choice's items: from the table, or for the resolution and the language, from the screen and the game.
- (void)fill:(NSPopUpButton *)pop for:(const Option *)o
{
    NSString *choices = !strcmp(o->env, "_display") ? display_choices() : !strcmp(o->env, "_language") ? language_choices(self.game) : S(o->choices);
    [pop removeAllItems];
    for (NSString *pair in [choices componentsSeparatedByString:@"|"]) {
        NSRange eq = [pair rangeOfString:@"="];
        if (eq.location == NSNotFound) continue;
        [pop addItemWithTitle:[pair substringFromIndex:eq.location + 1]];
        pop.lastItem.representedObject = [pair substringToIndex:eq.location];
    }
}

- (void)fillModules
{
    for (int i = 0; i < k_option_count; i++)
        if (!strcmp(k_options[i].env, "_language") && self.controls[@"_language"]) [self fill:(NSPopUpButton *)self.controls[@"_language"] for:&k_options[i]];
    [self.modules removeAllItems];
    if (!self.game) return;
    NSString *folder = [self.game stringByAppendingPathComponent:@"Modules"];
    NSArray<NSString *> *names = [[[NSFileManager defaultManager] contentsOfDirectoryAtPath:folder error:nil] sortedArrayUsingSelector:@selector(localizedCaseInsensitiveCompare:)];
    for (NSString *name in names) {
        BOOL dir = NO;
        if ([name hasPrefix:@"."] || ![[NSFileManager defaultManager] fileExistsAtPath:[folder stringByAppendingPathComponent:name] isDirectory:&dir] || !dir) continue;
        [self.modules addItemWithTitle:name];
    }
    NSString *current = [[NSString stringWithContentsOfFile:module_file() encoding:NSUTF8StringEncoding error:nil] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
    if (current.length && [self.modules itemWithTitle:current]) [self.modules selectItemWithTitle:current];
    else if ([self.modules itemWithTitle:@"Native"]) [self.modules selectItemWithTitle:@"Native"];
}

- (void)chooseGame:(id)sender
{
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    panel.canChooseDirectories = YES, panel.canChooseFiles = NO;
    panel.message = @"Select the \"MountBlade Warband\" folder (the one that contains \"Mount and Blade.app\").";
    if ([panel runModal] != NSModalResponseOK) return;
    NSString *chosen = panel.URL.path;
    if (!is_game(chosen)) {
        [self alert:@"That folder does not contain \"Mount and Blade.app\"."];
        return;
    }
    [[NSFileManager defaultManager] createDirectoryAtPath:support_dir() withIntermediateDirectories:YES attributes:nil error:nil];
    [chosen writeToFile:saved_game_path() atomically:YES encoding:NSUTF8StringEncoding error:nil];
    self.game = chosen;
    [self fillModules];
    [self refresh];
}

- (void)alert:(NSString *)text
{
    NSAlert *a = [NSAlert new];
    a.messageText = @"Warband Metal";
    a.informativeText = text;
    a.alertStyle = NSAlertStyleWarning;
    [a runModal];
}

// Replaces this process with the game, the library loaded into it.
- (void)play:(id)sender
{
    if (!self.game) {
        [self chooseGame:nil];
        if (!self.game) return;
    }
    NSString *library = [[NSBundle mainBundle] pathForResource:@"libwbmetal" ofType:@"dylib"];
    if (!library) {
        [self alert:@"The renderer library is missing from this copy of Warband Metal. Download the app again."];
        return;
    }
    [self save];
    write_game_settings(self.values);
    NSString *module = self.modules.titleOfSelectedItem;
    if (module.length && !getenv("WBM_MODULE")) { // the game reads the module's name from this file
        [[NSFileManager defaultManager] createDirectoryAtPath:module_file().stringByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:nil];
        [module writeToFile:module_file() atomically:YES encoding:NSUTF8StringEncoding error:nil];
    }
    NSDictionary<NSString *, NSString *> *env = [self environment];
    for (NSString *name in env) setenv(name.UTF8String, env[name].UTF8String, 1);
    setenv("DYLD_INSERT_LIBRARIES", library.fileSystemRepresentation, 1);
    NSString *binary = binary_in(self.game);
    chdir(binary.stringByDeletingLastPathComponent.fileSystemRepresentation); // steam_appid.txt is read from the working directory
    char *const argv[] = {(char *)binary.fileSystemRepresentation, NULL};
    execv(binary.fileSystemRepresentation, argv);
    [self alert:[NSString stringWithFormat:@"The game could not be started: %s", strerror(errno)]];
}

- (NSTextField *)label:(NSString *)text small:(BOOL)small
{
    NSTextField *t = [NSTextField labelWithString:text];
    t.font = [NSFont systemFontOfSize:small ? NSFont.smallSystemFontSize : NSFont.systemFontSize];
    if (small) t.textColor = NSColor.secondaryLabelColor;
    t.lineBreakMode = NSLineBreakByWordWrapping;
    t.maximumNumberOfLines = 2;
    [t setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow forOrientation:NSLayoutConstraintOrientationHorizontal];
    return t;
}

- (void)build
{
    self.controls = [NSMutableDictionary dictionary];
    self.readouts = [NSMutableDictionary dictionary];
    const CGFloat width = 700, control_width = 250;

    NSStackView *rows = [NSStackView new];
    rows.orientation = NSUserInterfaceLayoutOrientationVertical;
    rows.alignment = NSLayoutAttributeLeading;
    rows.spacing = 8;
    rows.edgeInsets = NSEdgeInsetsMake(14, 20, 14, 20);
    const char *group = "";
    for (int i = 0; i < k_option_count; i++) {
        const Option *o = &k_options[i];
        if (strcmp(group, o->group)) {
            group = o->group;
            NSTextField *heading = [NSTextField labelWithString:S(group)];
            heading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize + 1];
            [rows addArrangedSubview:heading];
            if (rows.arrangedSubviews.count > 1) [rows setCustomSpacing:18 afterView:rows.arrangedSubviews[rows.arrangedSubviews.count - 2]];
            if (is_game_setting(o)) {
                [rows setCustomSpacing:2 afterView:heading];
                [rows addArrangedSubview:[self label:@"The game's own settings: the presets leave them as they are." small:YES]];
            }
        }
        NSStackView *text = [NSStackView stackViewWithViews:@[[self label:S(o->title) small:NO], [self label:S(o->help) small:YES]]];
        text.orientation = NSUserInterfaceLayoutOrientationVertical;
        text.alignment = NSLayoutAttributeLeading;
        text.spacing = 1;
        [text.widthAnchor constraintEqualToConstant:width - control_width - 60].active = YES;

        NSView *side;
        NSControl *control;
        if (o->kind == kToggle) {
            control = [NSSwitch new];
            side = control;
        } else if (o->kind == kSlider) {
            NSSlider *slider = [NSSlider sliderWithValue:o->low minValue:o->low maxValue:o->high target:nil action:nil];
            slider.continuous = YES;
            NSTextField *readout = [NSTextField labelWithString:@""];
            readout.alignment = NSTextAlignmentRight;
            readout.font = [NSFont monospacedDigitSystemFontOfSize:NSFont.systemFontSize weight:NSFontWeightRegular];
            [readout.widthAnchor constraintEqualToConstant:38].active = YES;
            self.readouts[S(o->env)] = readout;
            control = slider;
            side = [NSStackView stackViewWithViews:@[slider, readout]];
        } else {
            NSPopUpButton *pop = [NSPopUpButton new];
            [self fill:pop for:o];
            control = pop;
            side = pop;
        }
        control.tag = i, control.target = self, control.action = @selector(changed:);
        self.controls[S(o->env)] = control;
        NSView *holder = [NSView new]; // the control sits at the right of a column of fixed width
        holder.translatesAutoresizingMaskIntoConstraints = NO;
        side.translatesAutoresizingMaskIntoConstraints = NO;
        [holder addSubview:side];
        [NSLayoutConstraint activateConstraints:@[
            [holder.widthAnchor constraintEqualToConstant:control_width], [side.trailingAnchor constraintEqualToAnchor:holder.trailingAnchor],
            [side.centerYAnchor constraintEqualToAnchor:holder.centerYAnchor], [holder.heightAnchor constraintGreaterThanOrEqualToAnchor:side.heightAnchor],
            o->kind == kToggle ? [side.widthAnchor constraintGreaterThanOrEqualToConstant:10] : [side.leadingAnchor constraintEqualToAnchor:holder.leadingAnchor constant:o->kind == kSlider ? 20 : 30]
        ]];
        NSStackView *row = [NSStackView stackViewWithViews:@[text, holder]];
        row.alignment = NSLayoutAttributeCenterY;
        row.spacing = 20;
        [rows addArrangedSubview:row];
    }

    NSScrollView *scroll = [NSScrollView new];
    scroll.hasVerticalScroller = YES;
    scroll.drawsBackground = NO;
    scroll.translatesAutoresizingMaskIntoConstraints = NO;
    NSClipView *clip = scroll.contentView;
    rows.translatesAutoresizingMaskIntoConstraints = NO;
    scroll.documentView = rows;
    [NSLayoutConstraint activateConstraints:@[[rows.leadingAnchor constraintEqualToAnchor:clip.leadingAnchor], [rows.trailingAnchor constraintEqualToAnchor:clip.trailingAnchor], [rows.topAnchor constraintEqualToAnchor:clip.topAnchor]]];

    // Above the list: the presets. Below it: the game, the module, and the button.
    self.presets = [NSPopUpButton new];
    for (int p = 0; p < k_preset_count; p++) [self.presets addItemWithTitle:k_presets[p]];
    [self.presets addItemWithTitle:@"My own choice"];
    self.presets.target = self, self.presets.action = @selector(presetChosen:);
    NSStackView *top = [NSStackView stackViewWithViews:@[[self label:@"Settings:" small:NO], self.presets]];

    self.modules = [NSPopUpButton new];
    self.gameLabel = [self label:@"" small:YES];
    self.gameLabel.maximumNumberOfLines = 1;
    self.gameLabel.lineBreakMode = NSLineBreakByTruncatingMiddle;
    NSButton *choose = [NSButton buttonWithTitle:@"Game folder…" target:self action:@selector(chooseGame:)];
    choose.controlSize = NSControlSizeSmall;
    choose.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
    NSButton *play = [NSButton buttonWithTitle:@"Play" target:self action:@selector(play:)];
    play.keyEquivalent = @"\r";
    play.controlSize = NSControlSizeLarge;
    [play.widthAnchor constraintGreaterThanOrEqualToConstant:110].active = YES;
    self.skip = [NSButton checkboxWithTitle:@"Start the game straight away next time (hold ⌥ Option while opening to see this window)" target:self action:@selector(skipChanged:)];
    self.skip.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
    NSView *gap = [NSView new];
    [gap setContentHuggingPriority:1 forOrientation:NSLayoutConstraintOrientationHorizontal];
    NSStackView *line = [NSStackView stackViewWithViews:@[[self label:@"Module:" small:NO], self.modules, gap, play]];
    NSStackView *where = [NSStackView stackViewWithViews:@[choose, self.gameLabel]];
    NSStackView *bottom = [NSStackView stackViewWithViews:@[line, where, self.skip]];
    bottom.orientation = NSUserInterfaceLayoutOrientationVertical;
    bottom.alignment = NSLayoutAttributeLeading;
    bottom.spacing = 8;
    [line.widthAnchor constraintEqualToAnchor:bottom.widthAnchor].active = YES;

    NSBox *rule1 = [NSBox new], *rule2 = [NSBox new];
    rule1.boxType = rule2.boxType = NSBoxSeparator;
    NSStackView *all = [NSStackView stackViewWithViews:@[top, rule1, scroll, rule2, bottom]];
    all.orientation = NSUserInterfaceLayoutOrientationVertical;
    all.alignment = NSLayoutAttributeLeading;
    all.spacing = 10;
    all.edgeInsets = NSEdgeInsetsMake(14, 0, 16, 0);
    all.translatesAutoresizingMaskIntoConstraints = NO;
    for (NSView *v in @[top, bottom]) {
        [v.leadingAnchor constraintEqualToAnchor:all.leadingAnchor constant:20].active = YES;
        [v.trailingAnchor constraintEqualToAnchor:all.trailingAnchor constant:-20].active = YES;
    }
    for (NSView *v in @[rule1, scroll, rule2]) [v.widthAnchor constraintEqualToAnchor:all.widthAnchor].active = YES;

    self.window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, width, 720)
                                              styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
    self.window.title = @"Warband Metal";
    self.window.contentView = [NSView new];
    [self.window.contentView addSubview:all];
    [NSLayoutConstraint activateConstraints:@[
        [all.leadingAnchor constraintEqualToAnchor:self.window.contentView.leadingAnchor], [all.trailingAnchor constraintEqualToAnchor:self.window.contentView.trailingAnchor],
        [all.topAnchor constraintEqualToAnchor:self.window.contentView.topAnchor], [all.bottomAnchor constraintEqualToAnchor:self.window.contentView.bottomAnchor],
        [self.window.contentView.widthAnchor constraintEqualToConstant:width]
    ]];
    self.window.contentMinSize = NSMakeSize(width, 420);
    [self.window center];
}

- (void)applicationDidFinishLaunching:(NSNotification *)note
{
    NSArray<NSString *> *args = NSProcessInfo.processInfo.arguments;
    [self load];
    self.game = find_game();
    if ([args containsObject:@"--print"]) {
        NSDictionary<NSString *, NSString *> *env = [self environment];
        for (NSString *name in [env.allKeys sortedArrayUsingSelector:@selector(compare:)]) printf("%s=%s\n", name.UTF8String, env[name].UTF8String);
        printf("game: %s\n", self.game ? self.game.UTF8String : "not found");
        for (int i = 0; i < k_option_count; i++)
            if (is_game_setting(&k_options[i])) { // as it goes into the game's files
                NSString *value = self.values[S(k_options[i].env)] ?: @"";
                if (!strncmp(k_options[i].env, "_disable_", 9)) value = value.intValue ? @"0" : @"1";
                printf("game %s = %s\n", k_options[i].env + 1, value.UTF8String);
            }
        exit(0);
    }
    NSUInteger snapshot = [args indexOfObject:@"--snapshot"];
    BOOL option = (NSEvent.modifierFlags & NSEventModifierFlagOption) != 0;
    [self build];
    [self fillModules];
    [self refresh];
    if (snapshot != NSNotFound && snapshot + 1 < args.count) {
        self.window.appearance = [NSAppearance appearanceNamed:NSAppearanceNameAqua]; // dark text, for the white the picture is drawn on
        [self.window.contentView layoutSubtreeIfNeeded];
        NSBitmapImageRep *rep = [self.window.contentView bitmapImageRepForCachingDisplayInRect:self.window.contentView.bounds];
        [self.window.contentView cacheDisplayInRect:self.window.contentView.bounds toBitmapImageRep:rep];
        [[rep representationUsingType:NSBitmapImageFileTypePNG properties:@{}] writeToFile:args[snapshot + 1] atomically:YES];
        exit(0);
    }
    if ([self.values[@"_skip"] isEqualToString:@"1"] && !option && self.game) [self play:nil];
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)app { return YES; }
- (void)applicationWillTerminate:(NSNotification *)note { [self save]; }

@end

int main(void)
{
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        app.activationPolicy = NSApplicationActivationPolicyRegular;
        NSMenu *bar = [NSMenu new], *menu = [NSMenu new];
        NSMenuItem *item = [NSMenuItem new];
        item.submenu = menu;
        [bar addItem:item];
        [menu addItemWithTitle:@"Quit Warband Metal" action:@selector(terminate:) keyEquivalent:@"q"];
        app.mainMenu = bar;
        static Launcher *launcher;
        launcher = [Launcher new];
        app.delegate = launcher;
        [app run];
    }
    return 0;
}
