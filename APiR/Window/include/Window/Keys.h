#pragma once
// Keys are custom for more consistent event handling due to fact that some windows has own and different key codes for
// same keys

// Some printable keys are mapped to actual key code
enum class KeyboardButton : unsigned int
{
  KEY_Q = 'q',
  KEY_W = 'w',
  KEY_E = 'e',
  KEY_R = 'r',
  KEY_T = 't',
  KEY_Y = 'y',
  KEY_U = 'u',
  KEY_I = 'i',
  KEY_O = 'o',
  KEY_P = 'p',
  KEY_A = 'a',
  KEY_S = 's',
  KEY_D = 'd',
  KEY_F = 'f',
  KEY_G = 'g',
  KEY_H = 'h',
  KEY_J = 'j',
  KEY_K = 'k',
  KEY_L = 'l',
  KEY_Z = 'z',
  KEY_X = 'x',
  KEY_C = 'c',
  KEY_V = 'v',
  KEY_B = 'b',
  KEY_N = 'n',
  KEY_M = 'm',

  KEY_1 = '1',
  KEY_2 = '2',
  KEY_3 = '3',
  KEY_4 = '4',
  KEY_5 = '5',
  KEY_6 = '6',
  KEY_7 = '7',
  KEY_8 = '8',
  KEY_9 = '9',
  KEY_0 = '0',

  KEY_SPACE = ' ',
  KEY_TAB = '\t',
  KEY_LEFT_SHIFT = 256,
  KEY_LEFT_CTRL = 257,
  KEY_LEFT_ALT = 258,
  KEY_RIGHT_SHIFT = 259,
  KEY_RIGHT_CTRL = 260,
  KEY_RIGHT_ALT = 261,
  KEY_ESCAPE = 262,
  KEY_CAPS = 263,
  KEY_ENTER = 264,

  KEY_F1 = 265,
  KEY_F2 = 266,
  KEY_F3 = 267,
  KEY_F4 = 268,
  KEY_F5 = 269,
  KEY_F6 = 270,
  KEY_F7 = 271,
  KEY_F8 = 272,
  KEY_F9 = 273,
  KEY_F10 = 274,
  KEY_F11 = 275,
  KEY_F12 = 276,

  KEY_UNDEFINED = '\0',
};

enum class MouseButton : unsigned char
{
  MOUSE_LEFT_BUTTON = 0,
  MOUSE_MIDDLE_BUTTON = 1,
  MOUSE_RIGHT_BUTTON = 2,
  MOUSE_UNDEFINED_BUTTON
};