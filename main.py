import pygame as pg
import sys
import random

# GAME SCREEN SIZE CONSTANT
WIDTH = 500
HEIGHT = 500


running = True
BACKGROUND= (234, 212, 252)
FALLING_COLOR = (0,233,255)

# player constants
PLAYER_COLOR = (255,24,24)
SIZE = 30
START_X = random.randrange(1,WIDTH)
START_Y = HEIGHT - SIZE - 10
pg.init()


# list to store falling objects
falling_object = []

# window setup
window = pg.display.set_mode((WIDTH,HEIGHT))
pg.display.set_caption("Dodge AI Dodge")
window.fill(BACKGROUND)
pg.display.flip()
fpsClock = pg.time.Clock()
FPS = 60



def create_falling_object():
   x = random.randint(0,WIDTH - 20)
   y = 0
   speed = random.randint(2,5)
   falling_object.append({"rect": pg.Rect(x,y,20,20),"speed":speed})

   
def update_falling_object():
   dirty_rect = []
   for obj in falling_object:
      old_rect = obj["rect"].copy()

      pg.draw.rect(window,BACKGROUND,old_rect)
      dirty_rect.append(old_rect)


      obj["rect"].y += obj["speed"] 
      pg.draw.rect(window,FALLING_COLOR,obj["rect"])
      dirty_rect.append(obj["rect"])


   falling_object[:] = [obj for obj in falling_object if obj["rect"].y < HEIGHT]
   return dirty_rect

def draw_ai_player(x,y):
   shape= pg.Rect(x,y,SIZE,SIZE)
   pg.draw.rect(window,PLAYER_COLOR,shape)
   
   pg.display.update(shape)
   return shape

def clean_ai_player(x,y):
   shape = pg.Rect(x,y,SIZE,SIZE)
   pg.draw.rect(window,BACKGROUND,shape)

def player_move_left(x,y):
   # x and y is current player location 
   new_x = x - 1
   if 0 > new_x:
      new_x = 0
   
   clean_ai_player(x,y)
   draw_ai_player(new_x,y)
   return new_x,y

 
def player_move_right(x,y):
   # x and y is current player location 

   new_x = x + 1

   if new_x + SIZE > WIDTH:
      new_x = WIDTH - SIZE
   clean_ai_player(x,y)
   draw_ai_player(x+1,y)
   return new_x,y

def collision_detect(player):
   for obj in falling_object:
      if player.colliderect(obj["rect"]):
         return True
   return False

def ai_move(player_x,player_y):
   danger_zone = 100
   move_direction= 0

   for obj in falling_object:
      if player_y - danger_zone < obj["rect"].y < player_y:
         obj_center  = obj["rect"].x + 10
         player_center = player_x + SIZE / 2

         if obj_center < player_center:
            move_direction = 1
         else:
            move_direction = -1
         break

   new_x = player_x + move_direction

   if new_x < 0:
      new_x = 0
   if new_x + SIZE > WIDTH:
      new_x = WIDTH - SIZE
   return new_x

def main():
   global running
   frame_count = 0
   player_x = START_X
   player_y = START_Y   

   while running:
      # list to hold all changes regions this frame.
      dirty_rect = []

      for event in pg.event.get():
         if event.type == pg.QUIT:
            running= False

      # collision Detection 




      if frame_count % 30 == 0:
            create_falling_object()
      
      dirty_rect.extend(update_falling_object())


      old_player_rect = clean_ai_player(player_x,player_y)
      dirty_rect.append(old_player_rect)

      player_x = ai_move(player_x,player_y)




      player =  draw_ai_player(player_x,player_y)
      dirty_rect.append(player)

      if collision_detect(player) == True:
         print("Collision detection")
         running = False
      pg.display.update(dirty_rect)
      fpsClock.tick(FPS)
      frame_count += 1


   pg.quit()
   sys.exit()




   



main()
