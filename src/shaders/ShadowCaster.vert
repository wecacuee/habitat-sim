/*
    This file is part of Magnum.

    Original authors — credit is appreciated but not required:

        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019 —
            Vladimír Vondruš <mosra@centrum.cz>
        2016 — Bill Robinson <airbaggins@gmail.com>
*/

uniform highp mat4 transformationMatrix;

in highp vec4 position;

void main() {
  gl_Position = transformationMatrix * position;
}