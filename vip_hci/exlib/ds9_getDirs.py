#!/usr/bin/env python
from __future__ import division, print_function
"""Get useful directories for Mac (X or Classic), unix
or modern versions of Windows. Defines:

PlatformName: one of 'mac', 'win' or 'unix'
(note that Mac includes both MacOS X and Carbon).

getAppDirs(inclNone = False):
    Return up to two paths: the user's private and shared application support directory.

    If a directory does not exist, its path is set to None.

    A typical return on English system with inclNone True is:
    - MacOS X: [/Users/<username>/Library/Application Support,
        /Library/Application Support]
    - Mac Classic: ?
    - unix: [None, None] (use PATH to find applications!)
    - Windows:  [None, C:\Program Files]


getAppSuppDirs(inclNone = False):
    Return up to two paths: the user's private and shared application support directory.

    If a directory does not exist, its path is set to None.

    A typical return on English system is:
    - MacOS X: [/Users/<username>/Library/Application Support,
        /Library/Application Support]
    - Mac Classic: ?
    - unix: [<the user's home directory>, None]
    - Windows: [C:\Documents and Settings\<username>\Application Data,
        C:\Documents and Settings\All Users\Application Data]

getDocsDir():
    Return the path to the user's documents directory.

    Return None if the directory does not exist.

    A typical return on English system is:
    - MacOS X: /Users/<username>/Documents
    - Mac Classic: ?
    - unix: (depends on the flavor of unix)
    - Windows: C:\Documents and Settings\<username>\My Documents

getHomeDir():
    (documented below)

getPrefsDirs(inclNone = False):
    Return up to two paths: the user's private and shared preferences directory.

    Return None if the directory does not exist.

    A typical return on English system is:
    - MacOS X: [/Users/<username>/Library/Preferences,
        /Library/Preferences]
    - Mac Classic: [System Folder:Preferences, ?None?]
    - unix: [<the user's home directory>, None]
    - Windows: [C:\Documents and Settings\<username>\Application Data,
        C:\Documents and Settings\All Users\Application Data]

getPrefsPrefix():
    (documented below)

History:
2004-02-03 ROwen
2004-12-21 Improved main doc string. No code changes.
2005-07-11 ROwen    Modified getAppSuppDirs to return None for nonexistent directories.
                    Added getDocsDir.
2005-09-28 ROwen    Changed getPrefsDir to getPrefsDirs.
                    Added getAppDirs.
2005-10-05 ROwen    Added inclNone argument to getXXXDirs functions.
                    Documented getAppDirs.
                    Improved test code.
2005-10-06 ROwen    Make sure unix getHomeDir can never return [None]
                    (which could happen on Windows with missing required modules).
2006-02-28 ROwen    Bug fix: getHomeDir did not work on Windows.
"""
__all__ = ["PlatformName", "getAppDirs", "getAppSuppDirs", "getDocsDir", "getHomeDir",
    "getPrefsDirs", "getPrefsPrefix"]

import os

PlatformName = None

try:
    # try Mac
    from .ds9_getMacDirs import getAppDirs, getAppSuppDirs, getDocsDir, getPrefsDirs
    PlatformName = 'mac'
except ImportError:
    # try Windows
    try:
        from .ds9_getWinDirs import getAppDirs, getAppSuppDirs, getDocsDir, getPrefsDirs
        PlatformName = 'win'
    except ImportError:
        # assume Unix
        PlatformName = 'unix'
        def getAppDirs(inclNone = False):
            # use PATH to find apps on unix
            if inclNone:
                return [None, None]
            else:
                return []

        def getAppSuppDirs(inclNone = False):
            return getPrefsDirs(inclNone = inclNone)

        def getDocsDir():
            return getHomeDir()

        def getPrefsDirs(inclNone = False):
            if inclNone:
                return [getHomeDir(), None]
            else:
                homeDir = getHomeDir()
                if homeDir is not None:
                    return [homeDir]
            return []

def getHomeDir():
    """Return the path to the user's home directory.

    Return None if the directory cannot be determined.

    A typical return on English system is:
    - MacOS X: /Users/<username>
    - Mac Classic: ?
    - unix: (depends on the flavor of unix)
    - Windows: C:\Documents and Settings\<username>
    """
    if PlatformName == 'win':
        return os.environ.get('USERPROFILE')
    else:
        return os.environ.get('HOME')

def getPrefsPrefix():
    """Return the usual prefix for the preferences file:
    '.' for unix, '' otherwise.
    """
    global PlatformName
    if PlatformName == 'unix':
        return '.'
    return ''


if __name__ == '__main__':
    print('PlatformName     = %r' % PlatformName)
    print('getHomeDir()     = %r' % getHomeDir())
    print('getPrefsPrefix() = %r' % getPrefsPrefix())
    print()
    for inclNone in (False, True):
        print('getAppDirs(%s)     = %r' % (inclNone, getAppDirs(inclNone)))
        print('getAppSuppDirs(%s) = %r' % (inclNone, getAppSuppDirs(inclNone)))
        print('getPrefsDirs(%s)   = %r' % (inclNone, getPrefsDirs(inclNone)))
    print('getDocsDir()         = %r' % getDocsDir())
