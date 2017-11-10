from __future__ import absolute_import, division, print_function
"""Utilities for finding standard Mac directories.

History:
2004-02-04 ROwen
2004-02-12 ROwen    Modified to use fsref.as_pathname() instead of Carbon.File.pathname(fsref).
2005-07-11 ROwen    Modified getAppSuppDirs to return None for nonexistent directories.
                    Removed doCreate argument from getAppSuppDirs, getDocsDir and getPrefsDir.
                    Added getDocsDir.
2005-09-27 ROwen    Changed getPrefsDir to getPrefsDirs.
                    Added getAppDirs.
                    Refactored to use getMacUserDir and getMacUserSharedDirs.
2005-10-05 ROwen    Added inclNone argument to getXXXDirs functions.
                    Modified getStandardDir to return None if dirType is None.
                    Added getAppDirs and getPrefsDirs to the test code.
                    Removed obsolete getPrefsDir.
"""
import Carbon.Folder, Carbon.Folders
import MacOS

def getStandardDir(domain, dirType, doCreate=False):
    """Return a path to the specified standard directory or None if not found.
    
    The path is in unix notation for MacOS X native python
    and Mac colon notation for Carbon python,
    i.e. the form expected by the os.path module.
    
    Inputs:
    - domain: one of the domain constants found in Carbon.Folders,
        such as kUserDomain, kLocalDomain or kSystemDomain.
    - dirType: one of the type constants found in Carbon.Folders,
        such as kPreferencesFolderType or kTrashFolderType.
        If dirType is None, then returns None.
    - doCreate: try to create the directory if it does not exist?
    """
    if dirType is None:
        return None
    try:
        fsref = Carbon.Folder.FSFindFolder(domain, dirType, doCreate)
        return fsref.as_pathname()
    except MacOS.Error:
        return None

def getMacUserSharedDirs(dirType, inclNone = False):
    """Return the path to the user and shared folder of a particular type.
    
    Inputs:
    - dirType   one of the Carbon.Folders constants
    - inclNone  if True, paths to missing folders are set to None;
                if False (the default) paths to missing folders are omitted
    """ 
    retDirs = []
    for domain in Carbon.Folders.kUserDomain, Carbon.Folders.kLocalDomain:
        path = getStandardDir(
            domain = domain,
            dirType = dirType,
            doCreate = False,
        )
        if (path is not None) or inclNone:
            retDirs.append(path)
    return retDirs

def getMacUserDir(dirType):
    """Return the path to the user folder of a particular type,
    or None if the directory does not exist.

    Inputs:
    - dirType   one of the Carbon.Folders constants
    """ 
    return getStandardDir(
        domain = Carbon.Folders.kUserDomain,
        dirType = dirType,
        doCreate = False,
    )

def getAppDirs(inclNone = False):
    """Return up to two paths: user's private and shared application directory.

    Inputs:
    - inclNone  if True, paths to missing folders are set to None;
                if False (the default) paths to missing folders are omitted
    """
    return getMacUserSharedDirs(Carbon.Folders.kApplicationsFolderType, inclNone = inclNone)
    
def getAppSuppDirs(inclNone = False):
    """Return up to two paths: the user's private and shared application support directory.
    
    Inputs:
    - inclNone  if True, paths to missing folders are set to None;
                if False (the default) paths to missing folders are omitted
    """
    return getMacUserSharedDirs(Carbon.Folders.kApplicationSupportFolderType, inclNone = inclNone)

def getDocsDir():
    """Return the path to the user's documents directory.
    
    Return None if the directory does not exist.
    """
    return getMacUserDir(Carbon.Folders.kDocumentsFolderType)

def getPrefsDirs(inclNone = False):
    """Return up to two paths: the user's local and shared preferences directory.
    
    Inputs:
    - inclNone  if True, paths to missing folders are set to None;
                if False (the default) paths to missing folders are omitted
    """
    return getMacUserSharedDirs(Carbon.Folders.kPreferencesFolderType, inclNone = inclNone)
