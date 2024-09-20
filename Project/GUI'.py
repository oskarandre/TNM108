import PySimpleGUI as sg

sg.theme('Dark Gray 2')  

layout = [[sg.Text('Movie rating predictor 2000')],
      [sg.Text('Genres', size=(18, 1)), sg.InputText()],
      [sg.Text('Actors', size=(18, 1)), sg.InputText()],
      [sg.Text('Director', size=(18, 1)), sg.InputText()],
      [sg.Text('Short description of plot', size=(18, 1)), sg.InputText()],
      [sg.Submit(), sg.Cancel()]]

window = sg.Window('Movie rating predictor 2000', layout)

event, values = window.read()
window.close()
genres_input, cast_input, Director_input, overview_input  = values[0], values[1], values[2] ,values[3]        # get the data from the values dictionary
print(genres_input, cast_input,Director_input,overview_input)


#sg.theme_previewer()