from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tempfile
import os

from sbpl.motion_primitives import load_motion_pritimives, mprim_folder, dump_motion_primitives, \
    assert_motion_primitives_equal


def test_motion_primitive_file_dumping():
    mprimtives = load_motion_pritimives(os.path.join(mprim_folder(), 'all_file.mprim'))

    tempdir = tempfile.mkdtemp()
    primitives_filename = os.path.join(tempdir, 'test.mprim')
    dump_motion_primitives(mprimtives, primitives_filename)

    mprimtives_loaded = load_motion_pritimives(primitives_filename)

    assert_motion_primitives_equal(mprimtives, mprimtives_loaded)


if __name__ == '__main__':
    test_motion_primitive_file_dumping()
