import tkinter as tk
from PIL import Image, ImageTk

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interface de commande")
        self.geometry("500x400")
        
        # Définir le fond de la fenêtre en blanc
        self.configure(bg='white')

        # Liste pour suivre si chaque bouton a une couleur sélectionnée
        self.boutons_colors_selected = [False] * 4  # Liste pour 4 boutons, initialisé à False (aucune couleur sélectionnée)

        # Charger et afficher le logo
        self.charger_logo()

        # Liste des couleurs disponibles
        self.couleurs = ["", "green", "red", "black", "vide"]  # "green" correctement conservé

        # Créer un frame principal pour contenir les boutons et les menus
        self.frame = tk.Frame(self, bg='white')  # Fond blanc pour le frame
        self.frame.pack(pady=20)

        # Ajouter le texte au-dessus des boutons dans le frame des boutons
        self.ajouter_instructions()

        # Créer les boutons avec leurs menus déroulants
        self.creer_boutons()

        # Ajouter un bouton "Lancer" en bas à droite
        self.ajouter_bouton_lancer()

    def charger_logo(self):
        """Charge et affiche le logo en haut à gauche."""
        try:
            # Remplacez 'chemin/vers/logo.png' par le chemin de votre image
            self.logo_image = Image.open(r'C:\Users\utilisateur\Desktop\interface\interface\img1.jpg')  # Chemin absolu
            self.logo_image = self.logo_image.resize((70, 70), Image.LANCZOS)  # Redimensionner si nécessaire
            self.logo = ImageTk.PhotoImage(self.logo_image)

            logo_label = tk.Label(self, image=self.logo, bg='white', borderwidth=0)  # Fond blanc et sans bordure
            logo_label.image = self.logo  # Garder une référence à l'image
            logo_label.pack(side=tk.TOP, anchor='nw', padx=10, pady=10)  # Placer en haut à gauche
        except Exception as e:
            print(f"Erreur lors du chargement de l'image: {e}")

    def ajouter_instructions(self):
        """Ajoute des instructions pour l'utilisateur juste avant les boutons."""
        label_instructions = tk.Label(self.frame, text="Sélectionnez votre commande", bg='white', font=('Arial', 12))
        label_instructions.grid(row=0, column=0, columnspan=2, pady=10)  # Le texte est centré dans le grid

    def creer_boutons(self):
        """Crée quatre boutons avec menus déroulants pour changer leur couleur."""
        for i in range(4):
            self.creer_bouton_avec_menu(i)

    def creer_bouton_avec_menu(self, i):
        """Crée un bouton avec un menu déroulant pour changer sa couleur."""
        # Créer un frame pour chaque bouton et son menu déroulant
        sous_frame = tk.Frame(self.frame, bg='white')  # Fond blanc pour le sous-frame
        sous_frame.grid(row=i+1, column=0, padx=(10, 0), pady=5, sticky="w")  # Décalé de 1 ligne pour ne pas superposer avec le texte

        # Créer un bouton
        bouton = tk.Button(sous_frame, text=f"Bouton {i+1}", bg='white')  # Fond blanc pour le bouton
        bouton.grid(row=0, column=0, padx=10)

        # Créer une variable StringVar pour le menu déroulant de couleur
        couleur_selectionnee = tk.StringVar(self)
        couleur_selectionnee.set(self.couleurs[0])  # Choix par défaut (vide)

        # Créer le menu déroulant
        menu_couleur = tk.OptionMenu(sous_frame, couleur_selectionnee, *self.couleurs,
                                     command=lambda couleur: self.changer_couleur(bouton, couleur, i))
        menu_couleur.grid(row=0, column=1)

    def changer_couleur(self, bouton, couleur, index):
        """Change la couleur du bouton sélectionné et met à jour l'état dans la liste."""
        if couleur == "vide":
            bouton.config(bg="SystemButtonFace")  # Réinitialiser à la couleur par défaut
            self.boutons_colors_selected[index] = False
        elif couleur == "":
            bouton.config(bg="white")  # Pas de changement, rester blanc
            self.boutons_colors_selected[index] = False
        else:
            bouton.config(bg=couleur)
            self.boutons_colors_selected[index] = True  # Marque le bouton comme ayant une couleur sélectionnée

    def ajouter_bouton_lancer(self):
        """Ajoute un bouton 'Lancer' en bas à droite."""
        bouton_lancer = tk.Button(self, text="Lancer", bg="#921210", fg="white", command=self.action_lancer)
        bouton_lancer.pack(side=tk.BOTTOM, anchor='se', padx=10, pady=10)

    def action_lancer(self):
        """Action exécutée lorsque le bouton 'Lancer' est cliqué."""


# Lancer l'application
if __name__ == "__main__":
    app = Application()
    app.mainloop()
