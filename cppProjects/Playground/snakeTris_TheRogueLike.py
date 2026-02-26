import pygame
import math
import time
import random
# Init

pygame.init()
WIDTH = 1000
HEIGHT = 600
gameWidth = 300
gameHeight = 600
BACKGROUND = (0, 0, 0)
COLOUR = (0,255,0)
totalPointsEarned = 0
minLength = 5
speed = 0
lives = 50
level = 1
score = 0
appleamnt = 4
ApplyGravityOrb = False
ImmunityLimit = 40000000
gracePeriod = 100
appleIncrease = 100
LineClearMultipler = 1
currentGrace=0
GameState = "Playing"
running = True
snakeBlocks = [[0,60,(0,255,0)],[0,40,(0,255,0)],[0,20,(0,255,0)],[0,0,(0,255,0)]]
CostOfUpgrades = [1000, 5000, 10000, 50000, 80000, 160000, 300000, 500000, 800000, 1500000, 3000000, 5000000]
TetrisBlocks = []
apple = []
appleselection = {
    0: 7,
    1: 1
}
applePreselect = [0,0,0,0,0,0,0,1]
directions = {
    "RIGHT": (20, 0),
    "LEFT": (-20, 0),
    "UP": (0, -20),
    "DOWN": (0, 20)
}
timestwospeed = False
GravityStrength = 1
snakeDirection = "RIGHT"
Fall = False
immunity = 0
colorPreset = [(255,128,0), (255,255,0), (128,255,0), (0,255,0), (0,255,128), (0,255,255), (0,128,255), (0,0,255), (128,0,255), (255,0,255), (255,0,128), (255,255,255), (255,128,128), (128,255,255)]
AppleColor = [(255,0,0), (128,0,128), (165,42,42)]
screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill(BACKGROUND)
pygame.display.set_caption("SnakeTris")

def Button(text, x, y, width, height, mouse, click, becomes):
    global GameState
                
    if x + width > mouse[0] > x and y + height > mouse[1] > y:
        pygame.draw.rect(screen, (200,200,200), (x, y, width, height))
        if click:
            GameState = "Playing" if GameState == becomes else becomes
            return
    else:
        pygame.draw.rect(screen, (100,100,100), (x, y, width, height))
    font = pygame.font.Font(None, 36)
    text = font.render(text, True, (255, 255, 255))
    screen.blit(text, (x + 10, y + 10))

def inBoundaries(x,y,width=gameWidth,height=gameHeight):
    return x >= 0 and x < width and y >= 0 and y < height

def TurnRight(direction):
    if(direction == "UP"):
        return "RIGHT"
    elif(direction == "RIGHT"):
        return "DOWN"
    elif(direction == "DOWN"):
        return "LEFT"
    elif(direction == "LEFT"):
        return "UP"
    
def TurnLeft(direction):
    if(direction == "UP"):
        return "LEFT"
    elif(direction == "RIGHT"):
        return "UP"
    elif(direction == "DOWN"):
        return "RIGHT"
    elif(direction == "LEFT"):
        return "DOWN"
    
def Overlap(list1, list2):
    for item1 in list1:
        for item2 in list2:
            if item1[0] == item2[0] and item1[1] == item2[1]:
                return True
    return False

def SpawnAnApple(type=0):
    while True:
        appls = [random.randint(0, (gameWidth//20)-1)*20, random.randint(0, (gameHeight//20)-1)*20]
        if(not Overlap([appls], TetrisBlocks) and not Overlap ([appls], snakeBlocks) and not Overlap ([appls], apple)):
            break
    apple.append((appls,type))

def Render():
    global totalPointsEarned
    font = pygame.font.Font(None, 36)
    for y in range(0, gameHeight, 20):
        pygame.draw.rect(screen, (100,100,100), (gameWidth, y, 20, 20))

    level = ((score / 1000) ** 0.5) // 1 + 1
    leveltext = pygame.font.Font(None, 36).render(f"Level: {level}", True, (255, 255, 255))
    text = font.render(f"Score: {score}", True, (255, 255, 255))
    lifes = font.render(f"Lives Left : {lives}", True, (255, 255, 255))
    totalpoints = font.render(f"Totalpointsearned : {totalPointsEarned}", True, (255,255,255))
    screen.blit(text, (gameWidth + 10, 10))
    screen.blit(lifes, (gameWidth + 10, 67))
    screen.blit(leveltext, (gameWidth + 10, 38))
    screen.blit(totalpoints, (gameWidth + 10, 100))

    for tetris in TetrisBlocks:
        pygame.draw.rect(screen, (tetris[2]), (tetris[0], tetris[1], 20, 20))
    for appls in apple:
        pygame.draw.rect(screen, (AppleColor[appls[1]]), (appls[0][0], appls[0][1], 20, 20))
    for snakeblocks in snakeBlocks:
        pygame.draw.rect(screen, (snakeblocks[2]), (snakeblocks[0], snakeblocks[1], 20, 20))
    pygame.draw.rect(screen, (255, 255, 255), (snakeBlocks[-1][0], snakeBlocks[-1][1], 20, 20))
    pygame.display.flip()
    time.sleep(0.01)

def RemovefromAppleselection(type):
    for a in applePreselect:
        if(a == type):
            applePreselect.remove(a)
            return
        
def CheckLineClears():
        global score, totalPointsEarned
            # Check for line clears
        linesCleared = 0
        for _ in range(30):
            for y in range(0, gameHeight + 160, 20):
                cnt = [block for block in TetrisBlocks if block[1] == y]
                if len(cnt) == (gameWidth // 20):
                    linesCleared += 1
                    score += 500
                    for block in cnt: 
                        TetrisBlocks.remove(block)
                    for block in TetrisBlocks:
                        if block[1] < y:
                            block[1] += 20
                        
        
        score += (linesCleared ** 2) * 1000
        totalPointsEarned += (linesCleared ** 2) * 1000

def SnakeReset():
    global snakeBlocks, snakeDirection
    snakeBlocks = [[0,20,(0,255,0)],[0,0,(0,255,0)]]
    snakeDirection = "RIGHT"
    
def ApplesCovered():
    for block in TetrisBlocks:
        for appls in apple:
            if(block[0] == appls[0][0] and block[1] == appls[0][1]):
                storedType = appls[1]
                apple.remove(appls)
                SpawnAnApple(storedType)

def InitButtons(mouse,click):
    Button("Menu", gameWidth + 30, HEIGHT - 50, 80, 40, mouse, click, "Menu")
    Button("Upgrades", gameWidth + 120, HEIGHT - 50, 140, 40, mouse, click, "Upgrades")

for i in range(appleamnt):
    SpawnAnApple(0)
    RemovefromAppleselection(0)

while running:
    screen.fill(BACKGROUND)
    TetrisBlocks.sort(key=lambda block: block[1], reverse=True)
    get = False
    mouse = pygame.mouse.get_pos()
    click = False

    # handle effects
    if ApplyGravityOrb and not Fall:  
        print("applying gravity")
        for _ in range(GravityStrength):
            for block in TetrisBlocks:
                if(not(block[1] >= gameHeight-20) and not(Overlap([(block[0], block[1] + 20)], TetrisBlocks))):
                        block[1]+=20
            screen.fill(BACKGROUND)
            Render()
        ApplyGravityOrb = False
        print("applied gravity")
    ApplesCovered()    
    hasTurned = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            click = True
        elif (event.type == pygame.KEYDOWN and not(get) and not(Fall)):
            if event.key == pygame.K_LEFT:
                if(snakeDirection != "RIGHT"):
                    snakeDirection = "LEFT"
                    hasTurned = True
            elif event.key == pygame.K_RIGHT and not hasTurned:
                if(snakeDirection != "LEFT"):
                    snakeDirection = "RIGHT"
                    hasTurned = True
            elif event.key == pygame.K_UP and not hasTurned:
                if(snakeDirection != "DOWN"):
                    snakeDirection = "UP"
                    hasTurned = True
            elif event.key == pygame.K_DOWN and not hasTurned:
                if(snakeDirection != "UP"):
                        snakeDirection = "DOWN"
                        hasTurned = True
            elif event.key == pygame.K_s:
                timestwospeed = not timestwospeed
            elif event.key == pygame.K_g and score >= CostOfUpgrades[GravityStrength-1] and GameState == "Upgrades":
                score -= CostOfUpgrades[GravityStrength-1]
                GravityStrength += 1
                time.sleep(0.2)
            elif event.key == pygame.K_b and score >= CostOfUpgrades[appleselection[1]-1] and GameState == "Upgrades":
                score -= CostOfUpgrades[appleselection[1]-1]
                applePreselect.append(1)
                appleselection[1] += 1
                time.sleep(0.2)
            elif event.key == pygame.K_l and score >= 10000 and GameState == "Upgrades":
                score -= 10000
                lives += 1
                time.sleep(0.2)
            elif event.key == pygame.K_p and score >= CostOfUpgrades[appleamnt + 1] and GameState == "Upgrades":
                score -= CostOfUpgrades[appleamnt + 1]
                appleamnt += 1
                SpawnAnApple(0)
                RemovefromAppleselection(0)
                time.sleep(0.2)
            elif event.key == pygame.K_e and score >= CostOfUpgrades[appleIncrease//100] and GameState == "Upgrades":   
                score -= CostOfUpgrades[appleIncrease//100]
                appleIncrease += 100
                time.sleep(0.2)
            elif event.key == pygame.K_r and score >= CostOfUpgrades[int(LineClearMultipler**0.5)+2] and GameState == "Upgrades":
                score -= CostOfUpgrades[int(LineClearMultipler**0.5)+2]
                LineClearMultipler += 1
                time.sleep(0.2)
            elif event.key == pygame.K_u:
                GameState = "Upgrades" if GameState != "Upgrades" else "Menu"
            elif event.key == pygame.K_SPACE and len(snakeBlocks) >= minLength and not ApplyGravityOrb:
                Fall = True
                print("fall")
                get = True
    InitButtons(mouse, click)
    print(GameState)
    if(GameState == "Playing"):
        # print("notpouas")

        #handle input

        if(not(Fall)):

            immunity += 1
            nextBlock = snakeBlocks[-1][:]
            nextBlock[0] += directions[snakeDirection][0]
            nextBlock[1] += directions[snakeDirection][1]
            
            if (nextBlock in snakeBlocks 
                or not(inBoundaries(nextBlock[0],nextBlock[1],gameWidth,gameHeight)) 
                or Overlap([nextBlock], TetrisBlocks)):
                if(len(snakeBlocks) >= minLength):
                    Fall = True
                    continue

                elif (immunity < ImmunityLimit):
                    snakeDirection = TurnLeft(TurnRight(snakeDirection))
                    continue

                else:
                    currentGrace += 1
                    if(currentGrace < gracePeriod):
                        continue
                    else:
                        currentGrace = 0

                    immunity = 0
                    lives -= 1
                    if(lives == -1):
                        print("WALLBANG")
                        print(lives)
                        running = False
                    else:
                        SnakeReset()
                        continue
                continue

            eat = False
            typeEaten = 0
            for appls in apple:
                if(nextBlock[0] == appls[0][0] and nextBlock[1] == appls[0][1]):
                        nextBlock[2] = random.choice(colorPreset)
                        snakeBlocks.append(nextBlock)
                        typeEaten = appls[1]
                        apple.remove(appls)
                        applePreselect.append(typeEaten)
                        eat = True
                        if(typeEaten == 1):
                            ApplyGravityOrb = True
                        typeChosen = random.choice(applePreselect)
                        SpawnAnApple(typeChosen)
                        RemovefromAppleselection(typeChosen)
                        score += appleIncrease
                        totalPointsEarned += appleIncrease

            if(not(eat)):
                snakeBlocks.pop(0)
                snakeBlocks.append(nextBlock)
                
            
        else:
            print("falling")
            while(True):
                legal = True
                for block in snakeBlocks:
                    if(block[1] >= gameHeight-20):
                        legal = False
                        break
                    if(Overlap([(block[0], block[1] + 20)], TetrisBlocks)):
                        legal = False
                        break
                if(legal):
                    for k in snakeBlocks:
                        k[1] += 20
                    
                    screen.fill(BACKGROUND)
                    InitButtons(mouse, click)
                    Render()
                else:
                    break    
            
            screen.fill(BACKGROUND)
            InitButtons(mouse, click)
            Render()
            TetrisBlocks.extend(snakeBlocks)
            CheckLineClears()
            if(len(TetrisBlocks) == 0):
                score += 3000 # all clear
            SnakeReset()
            Fall = False
        Render()    
        CheckLineClears()
        speed = min(0.1, 1000/(score+1))
        if(timestwospeed):
            speed /= 2
        time.sleep(speed)
    elif (GameState == "Menu"):
        # Render()
        font = pygame.font.Font(None, 36)
        ApplesTypes = [appls for appls in appleselection]
        if(0 in ApplesTypes):
            text = font.render("You can eat the red apples to gain points and grow longer!", True, (255, 255, 255))
            screen.blit(text, (10, HEIGHT//2 - 180))
            moretext = font.render(f"Your appledeck is currently made up of {appleselection[0]} red apples", True, (255, 255, 255))
            screen.blit(moretext, (10, HEIGHT//2 - 140))
        if(1 in ApplesTypes):
            text = font.render(f"The purple apples will apply gravity to all blocks!", True, (255, 255, 255))
            moretext = font.render(f"Gravity will cause all blocks to fall down by {GravityStrength} blocks!", True, (255, 255, 255))
            evenmoretext = font.render(f"Your appledeck is currently made up of {appleselection[1]} purple apples", True, (255, 255, 255))
            screen.blit(text, (10, HEIGHT//2 - 60))
            screen.blit(moretext, (10, HEIGHT//2 - 20))
            screen.blit(evenmoretext, (10, HEIGHT//2 + 20))
        
        namingtext = font.render("Welcome to SnakeTris The RogueLike!", True, (255, 255, 255))
        instructionstext = font.render("Eat apples to grow longer and earn points. (Press s for x2 speed)", True, (255, 255, 255))
        instructionstext2 = font.render("Press Space / Bonk into a wall to become a tetris piece! line clears also gives points", True, (255, 255, 255))
        screen.blit(namingtext, (10, HEIGHT//2 - 300))
        screen.blit(instructionstext, (10, HEIGHT//2 - 260))
        screen.blit(instructionstext2, (10, HEIGHT//2 - 220))
        for i in range(len(appleselection)):
            pygame.draw.rect(screen, AppleColor[i], (10, HEIGHT//2 + 60 + i*49, 20, 20))
            stuff = font.render(f"x {appleselection[i]}", True, (255, 255, 255))
            screen.blit(stuff, (40, HEIGHT//2 + 60 + i*40))
        pygame.display.flip()
    elif GameState == "Upgrades":
         font = pygame.font.Font(None, 36)
         text = font.render("Upgrades: (press U to open is ok)", True, (255, 255, 255))
         screen.blit(text, (10, HEIGHT//2 - 200))
         gravityupgrade = font.render(f"Gravity Strength : {GravityStrength} (Press G to upgrade for {CostOfUpgrades[GravityStrength-1]} Points)", True, (255, 255, 255))
         screen.blit(gravityupgrade, (10, HEIGHT//2 - 160))
         GravityAmountUpgrade = font.render(f"Gravity Orbs in Apple Deck : {appleselection[1]} (Press B to upgrade for {CostOfUpgrades[appleselection[1]-1]} Points)", True, (255, 255, 255))
         screen.blit(GravityAmountUpgrade, (10, HEIGHT//2 - 120))
         LifeUpgrade = font.render(f"Buy one life for (10000 points (press L))", True, (255, 255, 255))
         screen.blit(LifeUpgrade, (10, HEIGHT//2 - 80))
         appleamountupgrade = font.render(f"Apple Deck Preselection :{appleamnt} apples on screen (Press P to add one for {CostOfUpgrades[appleamnt + 1]})", True, (255, 255, 255))
         screen.blit(appleamountupgrade, (10, HEIGHT//2 - 40))
         EarningUpgradeApples = font.render(f"more points per apple! ({appleIncrease} points per apple. Press E to upgrade for {CostOfUpgrades[appleIncrease//100]} Points)", True, (255, 255, 255))
         screen.blit(EarningUpgradeApples, (10, HEIGHT//2))
         LineClearUpgrade = font.render(f"Increase line clear mult! ({LineClearMultipler}x points for line clears. Press R to upgrade for {CostOfUpgrades[int(LineClearMultipler**0.5)+2]} Points)", True, (255, 255, 255))
         screen.blit(LineClearUpgrade, (10, HEIGHT//2 + 40))
         pygame.display.flip()
print(f"Score = {score}")
