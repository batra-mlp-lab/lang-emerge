# script to handle easy table creation in HTML
import pdb, sys
import numpy as np

class HTML():
    def __init__(self, cols):
        self.template = '<html><head><style>'+\
                        'table#t01{width:100%; background-color:#fff}'+\
                        'table#t01 tr:nth-child(even){background-color:#ccc;}'\
                        +'table#t01 tr:nth-child(odd){background-color:#fff;}'\
                        +'table#t01 th{background-color:black;color:white}'+\
                        '</style></head><body><table id ="t01">'
        self.end = '</table></body></html>';
        self.content = '';
        self.rowContent = '<tr>'+'<td valign="top">%s</td>'*cols+'</tr>';
        self.imgTemplate = '<img src="%s" height="%d" width="%d"></img>';
        self.attTemplate = '<mark style="background-color:rgba(255,0,0,%f)"> %s </mark>|'

        # creating table
        self.numRows = None;
        self.numCols = cols;

    # Add a new row
    def addRow(self, *entries):
        # if first element is list, take it
        if type(entries[0]) == list: entries = entries[0];

        if len(entries) != self.numCols:
            print('Warning: Incompatible number of entries.\nTaking needed!')

        if len(entries) < self.numCols: # add 'null'
            for ii in xrange(self.numCols - len(entries)):
                entries.append('NULL')

        newRow = self.rowContent % tuple(entries);
        # Add newRow to content
        self.content += newRow;

    # setting the title
    def setTitle(self, titles):
        newTitles = [];
        for ii in titles: newTitles.append('<strong>%s</strong>' % ii);
        self.addRow(newTitles);

    # render and save page
    def savePage(self, filePath):
        pageContent = self.template + self.content + self.end;
        # allow new page and tab space
        pageContent = pageContent.replace('\n', '</br>');
        pageContent = pageContent.replace('\t', '&nbsp;'*10);
        with open(filePath, 'w') as fileId: fileId.write(pageContent);
        print('Written page to: %s' % filePath)

    # Return the string for an image
    def linkImage(self, imgPath, caption=None, height=100):
        # No caption provided
        if caption == None: return self.imgTemplate % (imgPath, height, height);

        string = 'Caption: %s</br>' % caption;
        return string + (self.imgTemplate % (imgPath, height, height));

    # display attention string
    def attentionRow(self, attWt):
        numQues = attWt.size;
        titles = ['Cap'];
        titles.extend(['%02d' % ii for ii in xrange(1, numQues)]);
        string = '</br>';

        maxAtt = np.max(attWt);
        for ii in xrange(0, numQues):
            string += self.attTemplate % (attWt[ii]/maxAtt, titles[ii]);#, attWt[ii]/maxAtt);

        return string;
