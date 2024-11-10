import os
import requests
import random
import time
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.core.audio import SoundLoader
from kivy.clock import Clock
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.core.window import Window
import copia
from asignarCartas import cargar_cartas
from Qlearning import get_state, select_action, update_Q, reward, inicializar_Q, guardar_q_table

class CartaImage(ButtonBehavior, RelativeLayout):
    def __init__(self, carta, app, index, **kwargs):
        super().__init__(**kwargs)
        self.carta = carta
        self.app = app
        self.index = index

        # Imagen de la carta
        self.carta_img = Image(source=carta.image, size_hint=(None, None), size=(100, 150))
        self.add_widget(self.carta_img)

        # Imagen del marco
        self.marco_img = Image(source=f'images/frames/{carta.color.lower()}.png', size_hint=(None, None), size=(100, 150))
        self.add_widget(self.marco_img)

        # Imagen del número
        self.numero_img = Image(source=f'images/numbers/{carta.numero}.png', size_hint=(None, None), size=(18, 18), pos_hint={'x': 0.05, 'y': 0.72})
        self.add_widget(self.numero_img)

        # Imagen del elemento
        self.elemento_img = Image(source=f'images/elements/{carta.elemento.lower()}.png', size_hint=(None, None), size=(22, 22), pos_hint={'x': 0.02, 'y': 0.55})
        self.add_widget(self.elemento_img)

    def on_press(self):
        print(f'Carta: Color={self.carta.color}, Número={self.carta.numero}, Elemento={self.carta.elemento}')
        self.app.carta_seleccionada(self)

    def actualizar_carta(self, nueva_carta, ruta_local):
        self.carta = nueva_carta
        self.carta_img.source = ruta_local
        self.carta_img.reload()

        # Actualizar el marco
        self.marco_img.source = f'images/frames/{nueva_carta.color.lower()}.png'
        self.marco_img.reload()

        # Actualizar el número
        self.numero_img.source = f'images/numbers/{nueva_carta.numero}.png'
        self.numero_img.reload()

        # Actualizar el elemento
        self.elemento_img.source = f'images/elements/{nueva_carta.elemento.lower()}.png'
        self.elemento_img.reload()

class CartaGrande(RelativeLayout):
    def __init__(self, carta, ruta_local, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (200, 300)
        
        # Imagen de la carta
        self.carta_img = Image(source=ruta_local, size_hint=(None, None), size=(200, 300))
        self.add_widget(self.carta_img)

        # Imagen del marco
        self.marco_img = Image(source=f'images/frames/{carta.color.lower()}.png', size_hint=(None, None), size=(200, 300))
        self.add_widget(self.marco_img)

        # Imagen del número
        self.numero_img = Image(source=f'images/numbers/{carta.numero}.png', size_hint=(None, None), size=(36, 36), pos_hint={'x': 0.05, 'y': 0.72})
        self.add_widget(self.numero_img)

        # Imagen del elemento
        self.elemento_img = Image(source=f'images/elements/{carta.elemento.lower()}.png', size_hint=(None, None), size=(44, 44), pos_hint={'x': 0.02, 'y': 0.55})
        self.add_widget(self.elemento_img)

class CardJitsu(App):
    def build(self):
        inicializar_Q()
        self.victorias = {"User": {"Fuego": [], "Agua": [], "Nieve": []}, "IA": {"Fuego": [], "Agua": [], "Nieve": []}}
        self.historial_acciones = []  # Agregar historial de acciones
        self.layout = FloatLayout()
        self.ia_selection_times = [] # Lista para almacenar los tiempos de selección de la IA

        # -------------------- Fondo --------------------
        bg_image = Image(source='images/dojo.png', allow_stretch=True, keep_ratio=False)
        self.layout.add_widget(bg_image)

        # -------------------- Musica --------------------
        self.sound = SoundLoader.load('music/cardJitsu.mp3')
        self.soundFinal = SoundLoader.load('music/final.mp3')

        if self.sound:
            self.sound.loop = True  # Hace que la música se repita continuamente
            self.sound.play()

        # -------------------- Animacion de los pinguinos --------------------
        self.img = Image(source='images/pinguinos/pinguinos1.png', allow_stretch=True, keep_ratio=False)
        self.layout.add_widget(self.img)  # Asegúrate de añadir este widget al layout

        self.images = [f'images/pinguinos/pinguinos{i}.png' for i in range(1, 8)]
        self.current_index = 0

        Clock.schedule_interval(self.next_image, 0.3)

        # -------------------- Mostrar Cartas --------------------
        self.mostrar_cartas()

        # -------------------- Mostrar Insignias --------------------
        self.insignias_layout_user = BoxLayout(orientation='horizontal', spacing=10, size_hint=(None, None), size=(200, 200), pos_hint={'right': 1, 'top': 1})
        self.insignias_layout_ia = BoxLayout(orientation='horizontal', spacing=10, size_hint=(None, None), size=(200, 200), pos_hint={'x': 0.01, 'top': 1})
        self.layout.add_widget(self.insignias_layout_user)
        self.layout.add_widget(self.insignias_layout_ia)

        self.insignias_user = {elemento: BoxLayout(orientation='vertical', spacing=5, size_hint=(None, None), size=(50, 200)) for elemento in ['Fuego', 'Agua', 'Nieve']}
        self.insignias_ia = {elemento: BoxLayout(orientation='vertical', spacing=5, size_hint=(None, None), size=(50, 200)) for elemento in ['Fuego', 'Agua', 'Nieve']}

        for elemento, layout in self.insignias_user.items():
            self.insignias_layout_user.add_widget(layout)

        for elemento, layout in self.insignias_ia.items():
            self.insignias_layout_ia.add_widget(layout)

        self.mostrar_insignias()

        return self.layout

    def next_image(self, dt):
        self.current_index = (self.current_index + 1) % len(self.images)
        self.img.source = self.images[self.current_index]

    def descargar_imagen(self, url, nombre_archivo):
        """Descarga una imagen desde una URL y la guarda localmente."""
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(nombre_archivo, 'wb') as out_file:
                out_file.write(response.content)
            return nombre_archivo
        return None

    def mostrar_cartas(self):
        mazo = cargar_cartas()  # Cargar el mazo de cartas
        self.mazo = mazo
        self.mano, mazo = copia.seleccionar_cartas_mano(mazo)  # Seleccionar 5 cartas para la mano
        self.mano_ia, self.mazo_ia = copia.seleccionar_cartas_mano(mazo)  # Seleccionar 5 cartas para la IA
        rutas_imagenes = copia.obtener_rutas_imagenes(self.mano)  # Obtener las rutas de las imágenes de las cartas

        #Directorio temporal para las imágenes descargadas
        temp_dir = 'temp_images'
        os.makedirs(temp_dir, exist_ok=True)

        self.cartas_imagenes = []  # Almacenar referencias a las imágenes de las cartas

        # Añadir imágenes de las cartas a la interfaz
        for i, carta in enumerate(self.mano):
            nombre_archivo = os.path.join(temp_dir, f'carta_{i}.png')
            ruta_local = self.descargar_imagen(carta.image, nombre_archivo)
            if ruta_local:
                carta_img = CartaImage(carta=carta, app=self, index=i, size_hint=(None, None), size=(100, 150),
                                       pos_hint={'x': 0.1 + i * 0.15, 'y': 0.05})
                # Actualizar la fuente de la imagen con la ruta local
                carta_img.carta_img.source = ruta_local
                carta_img.marco_img.source = f'images/frames/{carta.color.lower()}.png'
                carta_img.numero_img.source = f'images/numbers/{carta.numero}.png'
                carta_img.elemento_img.source = f'images/elements/{carta.elemento.lower()}.png'
                self.layout.add_widget(carta_img)
                self.cartas_imagenes.append(carta_img)

    def mostrar_insignias(self):
        for elemento in ['Fuego', 'Agua', 'Nieve']:
            self.insignias_user[elemento].clear_widgets()
            self.insignias_ia[elemento].clear_widgets()

            # Mostrar insignias de usuario
            for color in self.victorias['User'][elemento]:
                insignia = RelativeLayout(size_hint=(None, None), size=(50, 50))
                fondo = Image(source=f'images/icons/icon_{color.lower()}.png', size_hint=(None, None), size=(50, 50))
                icono = Image(source=f'images/elements/{elemento.lower()}.png', size_hint=(None, None), size=(35, 35), pos_hint={'center_x': 0.5, 'center_y': 0.5})
                insignia.add_widget(fondo)
                insignia.add_widget(icono)
                self.insignias_user[elemento].add_widget(insignia)

            # Mostrar insignias de IA
            for color in reversed(self.victorias['IA'][elemento]):
                insignia = RelativeLayout(size_hint=(None, None), size=(50, 50))
                fondo = Image(source=f'images/icons/icon_{color.lower()}.png', size_hint=(None, None), size=(50, 50))
                icono = Image(source=f'images/elements/{elemento.lower()}.png', size_hint=(None, None), size=(35, 35), pos_hint={'center_x': 0.5, 'center_y': 0.5})
                insignia.add_widget(fondo)
                insignia.add_widget(icono)
                self.insignias_ia[elemento].add_widget(insignia)

    def carta_seleccionada(self, carta_image):
        # Iniciar temporizador
        start_time = time.perf_counter()

        # Selección de acción (carta) para la IA usando Q-learning
        estado_actual = get_state(self.victorias, self.mano_ia, self.mazo, [], self.historial_acciones)
        carta_ia = select_action(estado_actual, self.mano_ia)

        # Detener el temporizador después de la selección
        end_time = time.perf_counter()
        selection_time = end_time - start_time
        print(f"Tiempo de selección de la IA: {selection_time:.8f} segundos")

        # Guardar el tiempo en la lista
        self.ia_selection_times.append(selection_time)

        # Quitar la carta seleccionada de la mano de la IA
        self.mano_ia.remove(carta_ia)

        carta_user = carta_image.carta
        self.mano.remove(carta_user)

        resultado = copia.determinar_ganador(carta_user, carta_ia, self.victorias)
        self.historial_acciones.append((carta_user.elemento, carta_ia.elemento, resultado))

        if resultado == "Empate":
            recompensa = reward(False, False)
        else:
            recompensa = reward(resultado == "User", resultado == "IA")
            
        print(f'Resultado: {resultado}')
        print(f'Carta IA: Color={carta_ia.color}, Número={carta_ia.numero}, Elemento={carta_ia.elemento}')

        # Mostrar las cartas seleccionadas en grande
        indice_carta_user = carta_image.index
        self.mostrar_cartas_seleccionadas(indice_carta_user, carta_user, carta_ia)

        # Programar la actualización de insignias después de 4 segundos
        Clock.schedule_once(lambda dt: self.actualizar_despues_de_mostrar(carta_image, carta_user, carta_ia, resultado, recompensa, estado_actual), 4)

    def mostrar_cartas_seleccionadas(self, indice_carta_user, carta_user, carta_ia):
        nombre_archivo_ia = os.path.join('temp_images', 'carta_ia_5.png')
        ruta_local_ia = self.descargar_imagen(carta_ia.image, nombre_archivo_ia)
        
        # Verificar si la imagen se descargó correctamente
        if ruta_local_ia:
            self.cartas_seleccionadas_layout = FloatLayout(size_hint=(None, None), size=(Window.width, Window.height))
            
            # Mostrar la carta de la IA a la izquierda
            carta_ia_grande = CartaGrande(carta=carta_ia, ruta_local=ruta_local_ia, pos_hint={'center_x': 0.3, 'center_y': 0.5})
            self.cartas_seleccionadas_layout.add_widget(carta_ia_grande)

            # Mostrar la carta del usuario a la derecha
            ruta_local_user = f'temp_images/carta_{indice_carta_user}.png'
            carta_user_grande = CartaGrande(carta=carta_user, ruta_local=ruta_local_user, pos_hint={'center_x': 0.7, 'center_y': 0.5})
            self.cartas_seleccionadas_layout.add_widget(carta_user_grande)

            self.layout.add_widget(self.cartas_seleccionadas_layout)

    def actualizar_despues_de_mostrar(self, carta_image, carta_user, carta_ia, resultado, recompensa, estado_actual):
        # Eliminar las cartas mostradas en grande
        self.layout.remove_widget(self.cartas_seleccionadas_layout)

        self.historial_acciones.append((carta_user.elemento, carta_ia.elemento, resultado)) # Añadir la acción y el resultado al historial
        copia.mostrar_victorias(self.victorias) # Imprimir las victorias actualizadas
        self.mostrar_insignias() # Actualizar insignias

        estado_siguiente = get_state(self.victorias, self.mano_ia, self.mazo, self.mazo_ia, self.historial_acciones)
        update_Q(estado_actual, carta_ia, recompensa, estado_siguiente, self.mano_ia)

        # Verificar condiciones de victoria
        ganador, victoria = copia.verificar_condicion_victoria(self.victorias)
        if ganador:
            self.mostrar_ganador(ganador, victoria)
            return  # Salir del método si hay un ganador

        # Reemplazar la carta seleccionada con una nueva carta del mazo
        if self.mazo:
            nueva_carta = random.choice(self.mazo)
            self.mazo.remove(nueva_carta)
            self.mano.append(nueva_carta)

            # Descargar la imagen de la nueva carta
            nombre_archivo = os.path.join('temp_images', f'carta_{self.cartas_imagenes.index(carta_image)}.png')
            ruta_local = self.descargar_imagen(nueva_carta.image, nombre_archivo)
            if ruta_local:
                # Actualizar la imagen y la carta en el widget
                carta_image.actualizar_carta(nueva_carta, ruta_local)

        # Reemplazar la carta seleccionada de la IA con una nueva carta del mazo
        if self.mazo_ia:
            nueva_carta_ia = random.choice(self.mazo_ia)
            self.mazo_ia.remove(nueva_carta_ia)
            self.mano_ia.append(nueva_carta_ia)

    def mostrar_ganador(self, ganador, victoria):
        # Calcular el promedio de los tiempos de selección de la IA
        if self.ia_selection_times:
            promedio_tiempo = sum(self.ia_selection_times) / len(self.ia_selection_times)
            print(f"Promedio de tiempo de selección de la IA: {promedio_tiempo:.8f} segundos")

            # Guardar el promedio en un archivo de texto sin sobrescribir los valores anteriores
            with open("ia_selection_times.txt", "a") as file:  # Modo "a" para append
                file.write(f"\n{promedio_tiempo:.8f}")

        guardar_q_table()  # Guarda resultados de la partida en la memoria de la IA

        # Detener la música de fondo
        if self.sound:
            self.sound.stop()

        # Reproducir la música final
        if self.soundFinal:
            self.soundFinal.play()

        # Crear un FloatLayout para contener el rectángulo y el botón
        ganador_layout = FloatLayout(size_hint=(None, None), size=(300, 150))

        # Obtener las dimensiones de la ventana
        window_width, window_height = Window.size
        # Calcular la posición para centrar el layout
        layout_x = (window_width - 300) / 2
        layout_x2 = (window_width - 180) / 2
        layout_y = (window_height - 150) / 2
        layout_y2 = (window_height - 90) / 2

        with ganador_layout.canvas:
            Color(0, 0.5, 1, 1)  # Color celeste
            rect = Rectangle(pos=(layout_x, layout_y), size=(300, 150))

        label = Label(text=f"{ganador} Wins", pos_hint={'center_x': 1.68, 'center_y': 2.8}, font_size=30, color=(1, 1, 1))
        ganador_layout.add_widget(label)

        with ganador_layout.canvas:
            Color(1, 1, 1, 1)  # Color blanco
            # Dibujar el rectángulo con bordes redondeados
            rect2 = RoundedRectangle(pos=(layout_x2, layout_y2), size=(180, 50), radius=[10])

        label2 = Label(text="OK", pos_hint={'center_x': 1.68, 'center_y': 2.35}, font_size=30, color=(0, 0, 0))
        ganador_layout.add_widget(label2)

        button = Button(pos=(layout_x2, layout_y2), size_hint=(None, None), size=(180, 50),
                        background_color=(0, 0, 0, 0))  # Fondo transparente
        button.bind(on_press=App.get_running_app().stop)  # Enlazar el evento on_press para cerrar la aplicación
        ganador_layout.add_widget(button)

        # Agregar el layout a la pantalla
        self.layout.add_widget(ganador_layout)

    def cerrar_juego(self, instance):
        self.ganador_popup.dismiss()  # Cierra el popup
        App.get_running_app().stop()  # Cierra la aplicación

if __name__ == '__main__':
    CardJitsu().run()
