from __future__ import absolute_import, division, print_function
"""Determine the application data directory for Windows.

Requires Mark Hammond's win32 extensions
<http://starship.python.net/crew/mhammond/win32/>.
Raises RuntimeError on import if win32 is not found.

Thanks to Mike C Fletcher for sample code showing how to use win32com.shell
and a pointer to Microsoft's documentation!

SHGetFolderPath is documented here:
<http://msdn.microsoft.com/library/default.asp?url=/library/en-us/shellcc/platform/shell/reference/functions/shgetfolderpath.asp>

The directory type constants are documented here:
<http://msdn.microsoft.com/library/default.asp?url=/library/en-us/shellcc/platform/shell/reference/enums/csidl.asp>

*** Note for py2exe ***

If you use py2exe to "freeze" an application using this code,
you must put the following into your setup.py script:

# The following code is necessary for a py2exe app to find win32com.shell.
# Solution from <http://starship.python.net/crew/theller/moin.cgi/WinShell>
import win32com
import py2exe.mf as modulefinder
for p in win32com.__path__[1:]:
    modulefinder.AddPackagePath("win32com", p)
for extra in ["win32com.shell"]:
    __import__(extra)
    m = sys.modules[extra]
    for p in m.__path__[1:]:
        modulefinder.AddPackagePath(extra, p)

History:
2004-02-04 ROwen
2005-07-11 ROwen    Modified getAppSuppDirs to return None for nonexistent directories.
                    Added getDocsDir.
2005-09-28 ROwen    Changed getPrefsDir to getPrefsDirs.
                    Added getAppDirs.
                    Removed unused import of _winreg
2005-09-30 ROwen    Raise ImportError (as getDirs expects), not RuntimeError
                    if run on non-windows system.
2005-10-05 ROwen    Bug fix: no shellcon. before  CSIDL_PROGRAM_FILES.
                    Added inclNone argument to getXXXDirs functions.
                    Modified getStandardDir to return None if dirType is None.
2005-10-06 ROwen    Modified to be compatible with py2exe (one needs trickery
                    to import win32com.shell).
2006-02-28 ROwen    Removed py2exe compatibility because it's more appropriate
                    for the py2exe setup.py script.
"""
import pywintypes
from win32com.shell import shell, shellcon

def getStandardDir(dirType):
    """Return a path to the specified standard directory or None if not found. 

    The path is in the form expected by the os.path module.

    Inputs:
    - dirType: one of CSID constants, as found in the win32com.shellcon module,
        such as CSIDL_APPDATA or CSIDL_COMMON_APPDATA.
        If dirType is None, then returns None.

    Note: in theory one can create the directory by adding CSILD_FLAG_CREATE
    to dirType, but in practice this constant is NOT defined in win32com.shellcon,
    so it is risky (you would have to specify an explicit integer and hope it did
    not change).
    """

    if dirType is None:
        return None
    try:
        return shell.SHGetFolderPath(
            0, # hwndOwner (0=NULL)
            dirType,
            0, # hToken (0=NULL, no impersonation of another user)
            0  # dwFlag: want SHGFP_TYPE_CURRENT but it's not in shellcon; 0 seems to work
        )
    except pywintypes.com_error:
        return None

def getAppDirs(inclNone = False):
    """Return up to two paths: the user's private and shared applications directory.
    On Windows only the shared one exists.

    Inputs:
    - inclNone  if True, paths to missing folders are set to None;
                if False (the default) paths to missing folders are omitted

    A typical return on English system with inclNone True is:
    [None
    C:\Program Files]
    """
    retDirs = []
    for dirType in (None, shellcon.CSIDL_PROGRAM_FILES):
        path = getStandardDir(dirType)
        if (path is not None) or inclNone:
            retDirs.append(path)
    return retDirs

def getAppSuppDirs(inclNone = False):
    """Return up to two paths: the user's private and shared application support directory.

    Inputs:
    - inclNone  if True, paths to missing folders are set to None;
                if False (the default) paths to missing folders are omitted

    A typical return on English system is:
    [C:\Documents and Settings\<username>\Application Data,
    C:\Documents and Settings\All Users\Application Data]
    """
    retDirs = []
    for dirType in (shellcon.CSIDL_APPDATA, shellcon.CSIDL_COMMON_APPDATA):
        path = getStandardDir(dirType)
        if (path is not None) or inclNone:
            retDirs.append(path)
    return retDirs

def getDocsDir():
    """Return the path to the user's documents directory.

    Return None if the directory does not exist.

    A typical result on an English system is:
    C:\Documents and Settings\<username>\My Documents
    """
    return getStandardDir(shellcon.CSIDL_PERSONAL)

def getPrefsDirs(inclNone = False):
    """Return up to two paths: the user's private and shared preferences directory.
    On Windows this is actually the application data directory.

    Inputs:
    - inclNone  if True, paths to missing folders are set to None;
                if False (the default) paths to missing folders are omitted

    A typical return on English system is:
    [C:\Documents and Settings\<username>\Application Data,
    C:\Documents and Settings\All Users\Application Data]
    """
    return getAppSuppDirs(inclNone = inclNone)


if __name__ == "__main__":
    print("Testing")
    for inclNone in (False, True):
        print('getAppDirs(%s)     = %r' % (inclNone, getAppDirs(inclNone)))
        print('getAppSuppDirs(%s) = %r' % (inclNone, getAppSuppDirs(inclNone)))
        print('getPrefsDirs(%s)   = %r' % (inclNone, getPrefsDirs(inclNone)))
    print('getDocsDir()         = %r' % getDocsDir())
